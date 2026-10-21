"""ReconciliationApp MVP 5.0 — faster reconciliation for larger CSV files."""
import io
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

import pandas as pd
import streamlit as st

CENT = Decimal('0.01')
STATUS_ORDER = ['PAID', 'PARTIALLY_PAID', 'UNPAID', 'OVERPAID']
REASONS = {
    'MISSING_INVOICE_ID': 'Missing invoice ID',
    'INVOICE_NOT_FOUND': 'Invoice not found',
    'PAYMENT_ID_CONFLICT': 'Conflicting payment ID',
    'DUPLICATE': 'Duplicate payment',
}


def parse_amount(value, context):
    try:
        number = Decimal(str(value).strip())
        if not number.is_finite() or number < 0:
            raise ValueError('Amount must be finite and non-negative')
        if number.as_tuple().exponent < -2 and number != number.quantize(CENT):
            raise ValueError('More than two decimal places')
        return number.quantize(CENT, rounding=ROUND_HALF_UP)
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError(f'{context}: invalid amount {value!r}. Use non-negative numbers with up to two decimal places.') from exc


def read_csv(upload, required, label):
    try:
        df = pd.read_csv(upload, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    except Exception as exc:
        raise ValueError(f'{label}: could not read CSV ({exc})') from exc
    df.columns = df.columns.str.strip()
    if df.columns.duplicated().any():
        raise ValueError(f'{label}: duplicate column names found.')
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f'{label}: missing required columns: {", ".join(missing)}')
    if df.empty:
        raise ValueError(f'{label}: CSV has no data rows.')
    for field in required:
        df[field] = df[field].astype(str).str.strip()
    for index, raw in enumerate(df['amount'], start=2):
        parse_amount(raw, f'{label} row {index}')
    return df


def reconcile(invoices, payments):
    """Reconcile using integer cents and vectorized pandas grouping.

    Exact duplicate payment IDs count once; conflicting IDs are all excluded.
    """
    invoices, payments = invoices.copy(), payments.copy()
    for label, df, key in [('Invoices', invoices, 'invoice_id'), ('Payments', payments, 'payment_id')]:
        if df[key].eq('').any():
            raise ValueError(f'{label}: {key} cannot be empty (first affected CSV row {df.index[df[key].eq("")][0] + 2}).')
    if invoices['customer'].eq('').any():
        raise ValueError('Invoices: customer cannot be empty.')
    if invoices['invoice_id'].duplicated().any():
        duplicate = invoices.loc[invoices['invoice_id'].duplicated(), 'invoice_id'].iloc[0]
        raise ValueError(f'Invoices: invoice_id {duplicate!r} appears more than once. Fix the source file.')

    # Validate exact monetary values; use integer cents to avoid floating-point drift.
    invoices['_cents'] = invoices['amount'].map(lambda x: int(parse_amount(x, 'Invoice') * 100))
    payments['_cents'] = payments['amount'].map(lambda x: int(parse_amount(x, 'Payment') * 100))
    if invoices['_cents'].sum() > 9_000_000_000_000_000 or payments['_cents'].sum() > 9_000_000_000_000_000:
        raise ValueError('Total amount is too large for this prototype.')

    # A payment ID is conflicting when its invoice ID or amount differs.
    signature_counts = payments.groupby('payment_id', sort=False)[['invoice_id', '_cents']].nunique(dropna=False)
    conflicting_ids = signature_counts.index[(signature_counts['invoice_id'] > 1) | (signature_counts['_cents'] > 1)]
    conflicts_mask = payments['payment_id'].isin(conflicting_ids)
    duplicates_mask = payments.duplicated('payment_id', keep='first') & ~conflicts_mask
    issues = payments.loc[conflicts_mask | duplicates_mask].drop(columns='_cents').copy()
    issues['reason'] = ['PAYMENT_ID_CONFLICT' if x else 'DUPLICATE' for x in conflicts_mask[conflicts_mask | duplicates_mask]]

    accepted_rows = payments.loc[~conflicts_mask & ~duplicates_mask].copy()
    missing_mask = accepted_rows['invoice_id'].str.lower().isin(['', 'none', 'nan', 'null'])
    unknown_mask = ~missing_mask & ~accepted_rows['invoice_id'].isin(invoices['invoice_id'])
    unmatched = accepted_rows.loc[missing_mask | unknown_mask].drop(columns='_cents').copy()
    unmatched['reason'] = ['MISSING_INVOICE_ID' if x else 'INVOICE_NOT_FOUND' for x in missing_mask[missing_mask | unknown_mask]]

    valid = accepted_rows.loc[~missing_mask & ~unknown_mask]
    paid_cents = valid.groupby('invoice_id', sort=False)['_cents'].sum()
    accepted = Decimal(int(valid['_cents'].sum())) / 100

    results = invoices[['invoice_id', 'customer']].copy()
    total = invoices['_cents'].to_numpy(dtype='int64')
    received = results['invoice_id'].map(paid_cents).fillna(0).to_numpy(dtype='int64')
    results['invoice_amount'] = total / 100
    results['paid_amount'] = received / 100
    results['remaining'] = (total - received).clip(min=0) / 100
    results['overpayment'] = (received - total).clip(min=0) / 100
    results['difference'] = abs(total - received) / 100
    # A zero-amount invoice with no payment is PAID, as in MVP 3.0.
    import numpy as np
    results['status'] = np.select(
        [(received == 0) & (total > 0), received < total, received == total],
        ['UNPAID', 'PARTIALLY_PAID', 'PAID'], default='OVERPAID'
    )
    columns = list(payments.drop(columns='_cents').columns) + ['reason']
    return (results, unmatched.reindex(columns=columns), issues.reindex(columns=columns), accepted)


def csv_data(frame):
    return frame.to_csv(index=False).encode('utf-8-sig')


def excel_data(sheets):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)
            ws = writer.sheets[name]
            ws.freeze_panes = 'A2'
            ws.auto_filter.ref = ws.dimensions
            for col_idx, name in enumerate(frame.columns, start=1):
                from openpyxl.utils import get_column_letter
                # Sample cells instead of traversing every Excel row a second time.
                samples = frame[name].head(200).astype(str)
                width = max(len(str(name)), samples.map(len).max() if not samples.empty else 0)
                ws.column_dimensions[get_column_letter(col_idx)].width = min(40, max(14, width + 3))
    return output.getvalue()


def euro(value):
    return f'€{value:,.2f}'


def main():
    st.set_page_config(page_title='ReconciliationApp | Workspace', page_icon='💼', layout='wide', initial_sidebar_state='expanded')
    st.markdown("""<style>
    .stApp {background: #f6f8fc; color: #14213d;}
    [data-testid="stSidebar"] {background: #101d38;}
    [data-testid="stSidebar"] * {color: #f2f5ff;}
    [data-testid="stSidebar"] [data-testid="stFileUploader"] {border-radius: 12px;}
    .block-container {padding-top: 1.8rem; max-width: 1500px;}
    .hero {background:linear-gradient(115deg,#122344,#224e88);padding:24px 28px;border-radius:18px;color:white;margin-bottom:18px;}
    .hero h1 {color:white;font-size:2rem;margin:0 0 6px;}
    .hero p {color:#dbe6fa;margin:0;}
    .section-title {font-size:1.35rem;font-weight:750;margin:18px 0 10px;}
    div[data-testid="stMetric"] {background:white;border:1px solid #e3e9f2;border-radius:14px;padding:18px 16px;box-shadow:0 2px 8px #0f24400a;}
    div[data-testid="stMetricLabel"] {color:#52627a;}
    div[data-testid="stMetricValue"] {color:#172a4d;}
    div[data-testid="stDataFrame"] {border-radius:12px;overflow:hidden;}
    </style>""", unsafe_allow_html=True)
    with st.sidebar:
        st.markdown('## 💼 ReconciliationApp')
        st.caption('FINANCE OPERATIONS · MVP 5.0')
        st.divider()
        page = st.radio('Workspace', ['Dashboard', 'Invoices', 'Exceptions', 'Reports'], label_visibility='collapsed')
        st.divider()
        st.markdown('#### Import data')
        invoices_file = st.file_uploader('Invoices CSV', type=['csv'], key='invoices')
        payments_file = st.file_uploader('Payments CSV', type=['csv'], key='payments')
        st.caption('Invoices: invoice_id, customer, amount')
        st.caption('Payments: payment_id, invoice_id, amount')
        st.divider()
        st.warning('Demo only. Use synthetic or non-sensitive data. Do not upload confidential customer records.')
    st.markdown('<div class="hero"><h1>Finance workspace</h1><p>Invoice & payment reconciliation · clearer visibility, faster decisions</p></div>', unsafe_allow_html=True)
    if invoices_file is None or payments_file is None:
        st.info('Upload both CSV files in the left sidebar to open your reconciliation workspace.')
        st.markdown('### Getting started')
        st.markdown('1. Upload invoices CSV.\n2. Upload payments CSV.\n3. Explore the dashboard, exceptions and reports.')
        return
    try:
        invoices = read_csv(invoices_file, ['invoice_id', 'customer', 'amount'], 'Invoices')
        payments = read_csv(payments_file, ['payment_id', 'invoice_id', 'amount'], 'Payments')
        results, unmatched, issues, accepted = reconcile(invoices, payments)
    except ValueError as exc:
        st.error(str(exc))
        return
    total_invoiced = sum((parse_amount(v, 'Invoice') for v in invoices['amount']), Decimal('0'))
    outstanding = Decimal(str(results['remaining'].sum())).quantize(CENT)
    overpaid = Decimal(str(results['overpayment'].sum())).quantize(CENT)
    status_counts = results['status'].value_counts().reindex(STATUS_ORDER, fill_value=0)
    if page == 'Dashboard':
        st.success('Reconciliation completed successfully.')
        st.markdown('<div class="section-title">Financial overview</div>', unsafe_allow_html=True)
        a,b,c,d = st.columns(4)
        a.metric('Total invoiced', euro(total_invoiced))
        b.metric('Accepted payments', euro(accepted))
        c.metric('Outstanding', euro(outstanding))
        d.metric('Overpayments', euro(overpaid))
        st.caption(f'{len(invoices):,} invoices · {len(payments):,} payment rows · {len(unmatched):,} unmatched · {len(issues):,} duplicate/conflict rows')
        st.markdown('<div class="section-title">Invoice health</div>', unsafe_allow_html=True)
        a,b,c,d = st.columns(4)
        for col, status, label in zip([a,b,c,d], STATUS_ORDER, ['Paid', 'Partially paid', 'Unpaid', 'Overpaid']):
            col.metric(label, f'{int(status_counts[status]):,}')
        left,right = st.columns([1.3,1])
        with left:
            st.markdown('#### Status distribution')
            st.bar_chart(status_counts.rename_axis('Status').rename('Invoices'))
        with right:
            st.markdown('#### Attention required')
            st.metric('Unmatched payment rows', f'{len(unmatched):,}')
            st.metric('Duplicate / conflict rows', f'{len(issues):,}')
            st.caption('Review exceptions before using figures for business decisions.')
        st.markdown('#### Recent invoices')
        st.dataframe(results.head(20), hide_index=True, use_container_width=True)
    elif page == 'Invoices':
        st.markdown('## Invoices')
        col1,col2=st.columns([1,2])
        with col1: statuses=st.multiselect('Filter by status', STATUS_ORDER)
        with col2: query=st.text_input('Search invoice or customer')
        filtered=results
        if statuses: filtered=filtered[filtered['status'].isin(statuses)]
        if query.strip():
            term=query.strip()
            filtered=filtered[filtered['invoice_id'].str.contains(term,case=False,regex=False) | filtered['customer'].str.contains(term,case=False,regex=False)]
        st.caption(f'Showing {len(filtered):,} of {len(results):,} invoices')
        st.dataframe(filtered,hide_index=True,use_container_width=True)
        st.download_button('Download all reconciliation results (CSV)',csv_data(results),'reconciliation_results.csv','text/csv')
    elif page == 'Exceptions':
        st.markdown('## Exceptions & payment review')
        left,right=st.columns(2)
        with left:
            st.markdown('#### Unmatched payments')
            if unmatched.empty: st.success('No unmatched payments.')
            else: st.dataframe(unmatched,hide_index=True,use_container_width=True)
            st.download_button('Download unmatched CSV',csv_data(unmatched),'unmatched_payments.csv','text/csv')
        with right:
            st.markdown('#### Duplicate / conflicting payments')
            if issues.empty: st.success('No duplicate or conflicting payments.')
            else: st.dataframe(issues,hide_index=True,use_container_width=True)
            st.download_button('Download issues CSV',csv_data(issues),'duplicates_conflicts.csv','text/csv')
        st.info('Identical duplicate payment IDs count once; conflicting IDs are excluded pending review.')
    else:
        st.markdown('## Reports & exports')
        st.caption('CSV export is generally faster and lighter for large datasets.')
        st.download_button('Download reconciliation CSV',csv_data(results),'reconciliation_results.csv','text/csv')
        st.download_button('Download unmatched CSV',csv_data(unmatched),'unmatched_payments.csv','text/csv')
        st.download_button('Download issues CSV',csv_data(issues),'duplicates_conflicts.csv','text/csv')
        if st.button('Prepare Excel workbook'):
            try:
                st.session_state['report_excel'] = excel_data({'Reconciliation':results,'Unmatched':unmatched,'Duplicates Conflicts':issues,'Invoices':invoices,'Payments':payments})
            except ImportError:
                st.warning('Excel export requires openpyxl.')
        if 'report_excel' in st.session_state:
            st.download_button('📥 Download complete Excel report',st.session_state['report_excel'],'reconciliation_report.xlsx','application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',type='primary')
        with st.expander('View uploaded data'):
            st.markdown('**Invoices**')
            st.dataframe(invoices,hide_index=True,use_container_width=True)
            st.markdown('**Payments**')
            st.dataframe(payments,hide_index=True,use_container_width=True)
    st.caption('Prototype · EUR only · financial results require review before operational use.')


if __name__ == '__main__':
    main()
