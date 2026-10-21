"""ReconciliationApp MVP 15.0 — collections and customer operations demo."""
import io
import hashlib
from datetime import date, datetime
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
    if label == 'Invoices':
        for col in ('invoice_date', 'due_date'):
            if col in df.columns:
                raw = df[col].astype(str).str.strip()
                parsed = pd.to_datetime(raw.where(raw.ne(''), None), format='%Y-%m-%d', errors='coerce')
                bad = raw.ne('') & parsed.isna()
                if bad.any():
                    raise ValueError(f'Invoices: invalid {col} at CSV row {int(bad[bad].index[0])+2}; use YYYY-MM-DD.')
                df[col] = raw
        if 'due_date' in df.columns and 'invoice_date' in df.columns:
            due = pd.to_datetime(df['due_date'].replace('', None), errors='coerce')
            issued = pd.to_datetime(df['invoice_date'].replace('', None), errors='coerce')
            invalid = due.notna() & issued.notna() & (due < issued)
            if invalid.any():
                raise ValueError(f'Invoices: due_date precedes invoice_date at CSV row {int(invalid[invalid].index[0])+2}.')
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
    for date_col in ('invoice_date', 'due_date'):
        if date_col in invoices.columns:
            results[date_col] = invoices[date_col].values
    results['status'] = np.select(
        [(received == 0) & (total > 0), received < total, received == total],
        ['UNPAID', 'PARTIALLY_PAID', 'PAID'], default='OVERPAID'
    )
    columns = list(payments.drop(columns='_cents').columns) + ['reason']
    return (results, unmatched.reindex(columns=columns), issues.reindex(columns=columns), accepted)


def customer_balances(results):
    """Aggregate invoice-level outstanding and overpayments per customer."""
    if results.empty:
        return pd.DataFrame(columns=['customer', 'invoices', 'total_invoiced', 'total_paid', 'outstanding', 'overpayments', 'unpaid_invoices'])
    summary = results.groupby('customer', as_index=False, sort=False).agg(
        invoices=('invoice_id', 'size'),
        total_invoiced=('invoice_amount', 'sum'),
        total_paid=('paid_amount', 'sum'),
        outstanding=('remaining', 'sum'),
        overpayments=('overpayment', 'sum'),
        unpaid_invoices=('status', lambda x: int(x.isin(['UNPAID', 'PARTIALLY_PAID']).sum())),
    )
    # Monetary inputs are exact cents; round aggregate floats for presentation.
    for name in ['total_invoiced', 'total_paid', 'outstanding', 'overpayments']:
        summary[name] = summary[name].round(2)
    return summary.sort_values(['outstanding', 'customer'], ascending=[False, True]).reset_index(drop=True)



def aging_report(results, as_of):
    """Invoice-level aging; no date is inferred when due_date is absent."""
    if 'due_date' not in results.columns:
        return None
    aged = results.copy()
    due = pd.to_datetime(aged['due_date'].replace('', None), format='%Y-%m-%d', errors='coerce')
    days = (pd.Timestamp(as_of).normalize() - due).dt.days
    aged['days_overdue'] = days.astype('Int64')
    aged['aging_bucket'] = 'NO_DUE_DATE'
    debt = aged['remaining'].gt(0)
    aged.loc[debt & due.notna() & days.le(0), 'aging_bucket'] = 'NOT_DUE'
    for lower, upper, label in [(1, 30, '1-30'), (31, 60, '31-60'), (61, 90, '61-90')]:
        aged.loc[debt & days.between(lower, upper), 'aging_bucket'] = label
    aged.loc[debt & days.gt(90), 'aging_bucket'] = '90+'
    aged.loc[~debt, 'aging_bucket'] = 'SETTLED'
    return aged


def aging_summary(aged):
    order = ['NOT_DUE', '1-30', '31-60', '61-90', '90+', 'NO_DUE_DATE']
    debt = aged.loc[aged['remaining'] > 0]
    summary = debt.groupby('aging_bucket')['remaining'].agg(['count', 'sum']).reindex(order, fill_value=0).reset_index()
    return summary.rename(columns={'count': 'invoices', 'sum': 'outstanding'}).round(2)


def payment_ledger(payments, unmatched, issues):
    """Label each raw payment row without claiming it was accepted when duplicated."""
    ledger = payments.copy().reset_index(drop=True)
    keys = ['payment_id', 'invoice_id', 'amount']
    if not set(keys).issubset(ledger.columns):
        return ledger
    # Conflict applies to all rows; identical repeats apply to rows after first.
    conflict_ids = set(issues.loc[issues['reason'].eq('PAYMENT_ID_CONFLICT'), 'payment_id'])
    ledger['review_status'] = 'ACCEPTED'
    ledger.loc[ledger['payment_id'].isin(conflict_ids), 'review_status'] = 'CONFLICT_EXCLUDED'
    duplicate = ledger.duplicated('payment_id', keep='first') & ~ledger['payment_id'].isin(conflict_ids)
    ledger.loc[duplicate, 'review_status'] = 'DUPLICATE_EXCLUDED'
    missing = ledger['invoice_id'].str.lower().isin(['', 'none', 'nan', 'null'])
    known = set(unmatched.loc[unmatched['reason'].eq('INVOICE_NOT_FOUND'), 'invoice_id'])
    active = ledger['review_status'].eq('ACCEPTED')
    ledger.loc[active & missing, 'review_status'] = 'MISSING_INVOICE_ID'
    ledger.loc[active & ledger['invoice_id'].isin(known), 'review_status'] = 'INVOICE_NOT_FOUND'
    return ledger


def readable_due(days, due_date, remaining):
    """Explain invoice due date in plain language without showing negative overdue days."""
    if remaining <= 0:
        return 'Settled'
    if pd.isna(days) or not str(due_date).strip():
        return 'Due date missing'
    days = int(days)
    if days < 0:
        return f'{abs(days)} days until due'
    if days == 0:
        return 'Due today'
    return f'{days} days overdue'


def collections_queue(results, as_of):
    """Build a deterministic, invoice-level collections list; no due date is invented."""
    aged = aging_report(results, as_of)
    if aged is None:
        queue = results.loc[results['remaining'] > 0].copy()
        queue['due_date'] = ''
        queue['days_overdue'] = pd.Series(pd.NA, index=queue.index, dtype='Int64')
        queue['aging_bucket'] = 'NO_DUE_DATE'
    else:
        queue = aged.loc[aged['remaining'] > 0].copy()
    queue['payment_timing'] = [readable_due(days, due, amount) for days, due, amount in
                               zip(queue['days_overdue'], queue['due_date'], queue['remaining'])]
    queue['priority'] = 'Review date'
    queue.loc[queue['aging_bucket'].eq('NOT_DUE'), 'priority'] = 'Upcoming'
    queue.loc[queue['aging_bucket'].eq('1-30'), 'priority'] = 'Medium'
    queue.loc[queue['aging_bucket'].isin(['31-60', '61-90', '90+']), 'priority'] = 'High'
    priority_order = {'High': 0, 'Medium': 1, 'Review date': 2, 'Upcoming': 3}
    queue['_priority_sort'] = queue['priority'].map(priority_order)
    return queue.sort_values(['_priority_sort', 'remaining', 'invoice_id'], ascending=[True, False, True]).drop(columns='_priority_sort').reset_index(drop=True)


def draft_reminder(customer, invoice_id, amount, due_date, days):
    """A draft only: no email address inferred and no message sent."""
    due_text = f'due on {due_date}' if str(due_date).strip() else 'with a due date to be confirmed'
    if pd.notna(days) and int(days) > 0:
        timing = f'Our records show that this invoice is {int(days)} days overdue.'
    elif pd.notna(days) and int(days) == 0:
        timing = 'Our records show that payment is due today.'
    elif pd.notna(days):
        timing = f'Our records show that payment is due in {abs(int(days))} days.'
    else:
        timing = 'Please confirm the payment due date with our team.'
    return (f'Subject: Payment follow-up — invoice {invoice_id}\n\n'
            f'Hello {customer},\n\n'
            f'We are following up on invoice {invoice_id}, {due_text}, '
            f'with an outstanding balance of {euro(amount)}. {timing}\n\n'
            'If you have already paid, please disregard this reminder and send us the payment reference. '
            'If you have any questions, please reply so we can help.\n\n'
            'Kind regards,\nAccounts Receivable Team')


def dataset_fingerprint(invoices_file, payments_file):
    digest = hashlib.sha256()
    for upload in (invoices_file, payments_file):
        data = upload.getvalue()
        digest.update(len(data).to_bytes(8, 'big'))
        digest.update(data)
    return digest.hexdigest()


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
    .stButton button[kind="primary"] {border-radius: 10px;}
    .section-title {font-size:1.35rem;font-weight:750;margin:18px 0 10px;}
    div[data-testid="stMetric"] {background:white;border:1px solid #e3e9f2;border-radius:14px;padding:18px 16px;box-shadow:0 2px 8px #0f24400a;}
    div[data-testid="stMetricLabel"] {color:#52627a;}
    div[data-testid="stMetricValue"] {color:#172a4d;}
    div[data-testid="stDataFrame"] {border-radius:12px;overflow:hidden;}
    </style>""", unsafe_allow_html=True)
    with st.sidebar:
        st.markdown('## 💼 ReconciliationApp')
        st.caption('FINANCE OPERATIONS · MVP 15.0')
        st.divider()
        page = st.radio('Workspace', ['Dashboard', 'Invoices', 'Customers', 'Collections', 'Payments', 'Aging', 'Exceptions', 'Reports'], label_visibility='collapsed')
        st.divider()
        st.markdown('#### Import data')
        invoices_file = st.file_uploader('Invoices CSV', type=['csv'], key='invoices')
        payments_file = st.file_uploader('Payments CSV', type=['csv'], key='payments')
        st.caption('Invoices: invoice_id, customer, amount · optional invoice_date, due_date (YYYY-MM-DD)')
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
    customers = customer_balances(results)
    ledger = payment_ledger(payments, unmatched, issues)
    fingerprint = dataset_fingerprint(invoices_file, payments_file)
    if st.session_state.get('collections_dataset') != fingerprint:
        st.session_state['collections_dataset'] = fingerprint
        st.session_state['collections_tasks'] = {}
        st.session_state['collections_activity'] = []
        st.session_state.pop('report_excel', None)
    queue = collections_queue(results, date.today())
    as_of = st.date_input('Reporting date', value=__import__('datetime').date.today(), key='as_of') if page == 'Aging' else __import__('datetime').date.today()
    aged = aging_report(results, as_of)
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
    elif page == 'Customers':
        st.markdown('## Customers & outstanding balances')
        st.caption('Grouped by customer name in the source CSV. Overpayments are never automatically netted against debts.')
        a, b, c = st.columns(3)
        a.metric('Customers', f'{len(customers):,}')
        b.metric('Total outstanding', euro(outstanding))
        c.metric('Customers with debt', f'{int(customers["outstanding"].gt(0).sum()):,}')
        search_col, debt_col = st.columns([2, 1])
        with search_col:
            customer_query = st.text_input('Search customer', key='customer_search')
        with debt_col:
            debt_only = st.checkbox('Show only customers with debt', value=False)
        visible = customers
        if customer_query.strip():
            visible = visible[visible['customer'].str.contains(customer_query.strip(), case=False, regex=False)]
        if debt_only:
            visible = visible[visible['outstanding'] > 0]
        st.caption(f'Showing {len(visible):,} of {len(customers):,} customers · highest outstanding first')
        st.dataframe(visible, hide_index=True, use_container_width=True)
        st.download_button('Download customer balances (CSV)', csv_data(visible), 'customer_balances.csv', 'text/csv')
        st.markdown('#### Customer details')
        if visible.empty:
            st.info('No customers match these filters.')
        else:
            selected_customer = st.selectbox('Select a customer', visible['customer'].tolist(), key='customer_detail')
            row = customers.loc[customers['customer'].eq(selected_customer)].iloc[0]
            a,b,c = st.columns(3)
            a.metric('Invoices', f'{int(row["invoices"]):,}')
            b.metric('Outstanding', euro(row['outstanding']))
            c.metric('Overpayments', euro(row['overpayments']))
            st.dataframe(results.loc[results['customer'].eq(selected_customer)], hide_index=True, use_container_width=True)
            st.caption('Customer identities are matched by name only. This prototype has no customer master-data IDs.')
        st.markdown('#### Largest outstanding balances')
        top = visible.loc[visible['outstanding'] > 0].head(15)
        if not top.empty:
            st.bar_chart(top.set_index('customer')['outstanding'])

    elif page == 'Collections':
        st.markdown('## Collections workspace')
        st.caption('Prioritize unpaid invoices, record follow-up tasks, and prepare reminder drafts. Nothing is sent automatically.')
        a,b,c = st.columns(3)
        a.metric('Open invoices', f'{len(queue):,}')
        a_high = queue['priority'].eq('High')
        b.metric('High-priority invoices', f'{int(a_high.sum()):,}')
        c.metric('High-priority balance', euro(queue.loc[a_high, 'remaining'].sum()))
        with st.expander('How priorities are assigned'):
            st.write('High: 31+ days overdue. Medium: 1–30 days overdue. Upcoming: not yet due. Review date: due date missing. These are demo rules, not credit-risk scores.')
        search, priority = st.columns([2, 1])
        with search:
            q = st.text_input('Search invoice or customer', key='collections_search')
        with priority:
            selected_priorities = st.multiselect('Priority', ['High', 'Medium', 'Upcoming', 'Review date'], key='collections_priority')
        visible = queue
        if q.strip():
            visible = visible[visible['invoice_id'].str.contains(q.strip(), case=False, regex=False) |
                              visible['customer'].str.contains(q.strip(), case=False, regex=False)]
        if selected_priorities:
            visible = visible[visible['priority'].isin(selected_priorities)]
        shown_cols = ['invoice_id', 'customer', 'remaining', 'due_date', 'payment_timing', 'priority', 'status']
        st.caption(f'Showing {len(visible):,} of {len(queue):,} open invoices')
        st.dataframe(visible[shown_cols], hide_index=True, use_container_width=True)
        st.download_button('Download collections queue CSV', csv_data(visible[shown_cols]), 'collections_queue.csv', 'text/csv')
        st.markdown('#### Follow-up and reminder draft')
        if visible.empty:
            st.info('No invoices match these filters.')
        else:
            invoice_ids = visible['invoice_id'].tolist()
            selected_invoice = st.selectbox('Choose an invoice', invoice_ids, key='collections_invoice')
            selected_row = queue.loc[queue['invoice_id'].eq(selected_invoice)].iloc[0]
            st.info(f'{selected_row["customer"]} · {euro(selected_row["remaining"])} outstanding · {selected_row["payment_timing"]}')
            task_states = ['Not started', 'Contact planned', 'Contacted', 'Promise to pay', 'Disputed', 'Resolved']
            saved = st.session_state['collections_tasks'].get(selected_invoice, {})
            with st.form('followup_form'):
                task_state = st.selectbox('Follow-up status', task_states,
                                          index=task_states.index(saved.get('status', 'Not started')))
                next_action = st.date_input('Next follow-up date', value=date.fromisoformat(saved['next_action']) if saved.get('next_action') else date.today())
                note = st.text_area('Internal note (demo only)', value=saved.get('note', ''), max_chars=1000,
                                    help='Use non-sensitive test notes only. Notes are not permanently stored.')
                save = st.form_submit_button('Save follow-up in this session', type='primary')
            if save:
                st.session_state['collections_tasks'][selected_invoice] = {
                    'status': task_state, 'next_action': next_action.isoformat(), 'note': note,
                }
                st.session_state['collections_activity'].append({
                    'recorded_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'invoice_id': selected_invoice, 'customer': selected_row['customer'],
                    'status': task_state, 'next_action': next_action.isoformat(), 'note': note,
                })
                st.success('Follow-up saved for this browser session. It is not stored in a database.')
            with st.expander('Prepare an email reminder (draft only)'):
                draft = draft_reminder(selected_row['customer'], selected_invoice,
                                       selected_row['remaining'], selected_row['due_date'], selected_row['days_overdue'])
                st.text_area('Copy and review before sending', value=draft, height=240, key=f'draft_{selected_invoice}')
                st.caption('No email is sent. Confirm amounts, recipient identity and due date before using the text.')
        st.markdown('#### Follow-up activity (this session only)')
        activity = st.session_state['collections_activity']
        if activity:
            history = pd.DataFrame(activity[::-1])
            st.dataframe(history, hide_index=True, use_container_width=True)
            st.download_button('Download session activity CSV', csv_data(history), 'followup_activity.csv', 'text/csv')
        else:
            st.info('No follow-ups recorded yet.')
        st.warning('Demo only: tasks and notes are held in temporary Streamlit session memory. They can disappear after refresh, restart or session expiry. Do not use for real collection operations.')
    elif page == 'Payments':
        st.markdown('## Payments ledger')
        st.caption('All source payment rows are shown. Excluded duplicates and conflicts are not counted in accepted payments.')
        q = st.text_input('Search payment ID or invoice ID', key='payment_query')
        statuses = sorted(ledger['review_status'].unique().tolist())
        selected = st.multiselect('Payment review status', statuses, key='payment_status')
        visible = ledger
        if selected:
            visible = visible.loc[visible['review_status'].isin(selected)]
        if q.strip():
            match = visible['payment_id'].str.contains(q.strip(), case=False, regex=False) | visible['invoice_id'].str.contains(q.strip(), case=False, regex=False)
            visible = visible.loc[match]
        st.caption(f'Showing {len(visible):,} of {len(ledger):,} payment rows')
        st.dataframe(visible, hide_index=True, use_container_width=True)
        st.download_button('Download payments ledger CSV', csv_data(visible), 'payments_ledger.csv', 'text/csv')
    elif page == 'Aging':
        st.markdown('## Receivables aging')
        if aged is None:
            st.info('To enable aging, add a due_date column to invoices.csv (YYYY-MM-DD). Missing dates are not estimated.')
        else:
            aging = aging_summary(aged)
            overdue = aging.loc[aging['aging_bucket'].isin(['1-30', '31-60', '61-90', '90+']), 'outstanding'].sum()
            a,b,c = st.columns(3)
            a.metric('Outstanding', euro(outstanding))
            b.metric('Overdue as of selected date', euro(overdue))
            c.metric('Invoices missing due date', f'{int(((aged["remaining"] > 0) & aged["due_date"].eq("")).sum()):,}')
            st.dataframe(aging, hide_index=True, use_container_width=True)
            st.bar_chart(aging.set_index('aging_bucket')['outstanding'])
            st.markdown('#### Invoice aging detail')
            detail = aged.loc[aged['remaining'] > 0].copy()
            detail['payment_timing'] = [readable_due(days, due, amount) for days, due, amount in
                                        zip(detail['days_overdue'], detail['due_date'], detail['remaining'])]
            display_cols = [col for col in detail.columns if col != 'days_overdue']
            st.dataframe(detail[display_cols], hide_index=True, use_container_width=True)
            st.caption('Negative days are shown as “days until due”; positive days as “days overdue”. The exported CSV retains numeric days_overdue for audit.')
            st.download_button('Download aging CSV', csv_data(aged), 'receivables_aging.csv', 'text/csv')
            st.caption('Aging uses invoice due dates, not payment transaction dates. Overpayments are not netted against debts.')
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
        st.download_button('Download customer balances CSV',csv_data(customers),'customer_balances.csv','text/csv')
        st.download_button('Download payment ledger CSV',csv_data(ledger),'payments_ledger.csv','text/csv')
        st.download_button('Download collections queue CSV', csv_data(queue), 'collections_queue.csv', 'text/csv')
        if st.session_state['collections_activity']:
            st.download_button('Download follow-up activity CSV', csv_data(pd.DataFrame(st.session_state['collections_activity'])), 'followup_activity.csv', 'text/csv')
        if aged is not None:
            st.download_button('Download aging CSV',csv_data(aged),'receivables_aging.csv','text/csv')
        if st.button('Prepare Excel workbook'):
            try:
                st.session_state['report_excel'] = excel_data(dict({'Reconciliation':results,'Unmatched':unmatched,'Duplicates Conflicts':issues,'Customer Balances':customers,'Payment Ledger':ledger,'Collections':queue,'Invoices':invoices,'Payments':payments}, **({'Aging':aged} if aged is not None else {})))
                st.session_state['report_signature'] = (fingerprint, str(as_of))
            except ImportError:
                st.warning('Excel export requires openpyxl.')
        if 'report_excel' in st.session_state and st.session_state.get('report_signature') == (fingerprint, str(as_of)):
            st.download_button('📥 Download complete Excel report',st.session_state['report_excel'],'reconciliation_report.xlsx','application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',type='primary')
        with st.expander('View uploaded data'):
            st.markdown('**Invoices**')
            st.dataframe(invoices,hide_index=True,use_container_width=True)
            st.markdown('**Payments**')
            st.dataframe(payments,hide_index=True,use_container_width=True)
    st.caption('Prototype · EUR only · no account isolation, database or secure storage · use synthetic/non-sensitive files only. Review all figures and reminder drafts before use.')


if __name__ == '__main__':
    main()
