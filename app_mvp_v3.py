"""ReconciliationApp MVP 3.0 — local CSV invoice/payment reconciliation."""
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
    invoices, payments = invoices.copy(), payments.copy()
    for label, df, key in [('Invoices', invoices, 'invoice_id'), ('Payments', payments, 'payment_id')]:
        if df[key].eq('').any():
            raise ValueError(f'{label}: {key} cannot be empty (first affected CSV row {df.index[df[key].eq("")][0] + 2}).')
    if invoices['customer'].eq('').any():
        raise ValueError('Invoices: customer cannot be empty.')
    if invoices['invoice_id'].duplicated().any():
        duplicate = invoices.loc[invoices['invoice_id'].duplicated(), 'invoice_id'].iloc[0]
        raise ValueError(f'Invoices: invoice_id {duplicate!r} appears more than once. Fix the source file.')

    totals = {r['invoice_id']: parse_amount(r['amount'], 'Invoice') for _, r in invoices.iterrows()}
    paid = {iid: Decimal('0.00') for iid in totals}
    groups = {}
    for _, row in payments.iterrows():
        groups.setdefault(row['payment_id'], []).append(row)

    issues, rejected = [], set()
    for pid, rows in groups.items():
        signatures = {(r['invoice_id'], parse_amount(r['amount'], 'Payment')) for r in rows}
        if len(signatures) > 1:
            rejected.add(pid)
            issues.extend({**r.to_dict(), 'reason': 'PAYMENT_ID_CONFLICT'} for r in rows)
        elif len(rows) > 1:
            issues.extend({**r.to_dict(), 'reason': 'DUPLICATE'} for r in rows[1:])

    unmatched, seen, accepted = [], set(), Decimal('0.00')
    for _, row in payments.iterrows():
        pid, iid = row['payment_id'], row['invoice_id']
        if pid in rejected or pid in seen:
            continue
        seen.add(pid)
        if not iid or iid.lower() in {'none', 'nan', 'null'}:
            unmatched.append({**row.to_dict(), 'reason': 'MISSING_INVOICE_ID'})
        elif iid not in totals:
            unmatched.append({**row.to_dict(), 'reason': 'INVOICE_NOT_FOUND'})
        else:
            amount = parse_amount(row['amount'], 'Payment')
            paid[iid] += amount
            accepted += amount

    results = []
    for _, row in invoices.iterrows():
        iid = row['invoice_id']
        total, received = totals[iid], paid[iid]
        status = ('UNPAID' if received == 0 and total > 0 else
                  'PARTIALLY_PAID' if received < total else
                  'PAID' if received == total else 'OVERPAID')
        results.append({
            'invoice_id': iid, 'customer': row['customer'],
            'invoice_amount': float(total), 'paid_amount': float(received),
            'remaining': float(max(total - received, Decimal('0'))),
            'overpayment': float(max(received - total, Decimal('0'))),
            'difference': float(abs(total - received)), 'status': status,
        })
    columns = list(payments.columns) + ['reason']
    return (pd.DataFrame(results), pd.DataFrame(unmatched, columns=columns),
            pd.DataFrame(issues, columns=columns), accepted)


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
            for cells in ws.columns:
                letter = cells[0].column_letter
                ws.column_dimensions[letter].width = min(40, max(14, max(len(str(c.value or '')) for c in cells) + 3))
    return output.getvalue()


def euro(value):
    return f'€{value:,.2f}'


def main():
    st.set_page_config(page_title='ReconciliationApp', page_icon='💼', layout='wide')
    st.title('💼 ReconciliationApp')
    st.caption('Payment reconciliation workspace · MVP 3.0')
    with st.sidebar:
        st.header('1 · Upload CSV files')
        invoices_file = st.file_uploader('Invoices CSV', type=['csv'], key='invoices')
        payments_file = st.file_uploader('Payments CSV', type=['csv'], key='payments')
        st.caption('Invoices: invoice_id, customer, amount')
        st.caption('Payments: payment_id, invoice_id, amount')
        st.divider()
        st.info('Prototype: use synthetic or non-sensitive test data. Do not upload confidential customer records to a public deployment.')

    if invoices_file is None or payments_file is None:
        st.info('Upload both CSV files in the left sidebar to see the reconciliation dashboard.')
        st.stop()
    try:
        invoices = read_csv(invoices_file, ['invoice_id', 'customer', 'amount'], 'Invoices')
        payments = read_csv(payments_file, ['payment_id', 'invoice_id', 'amount'], 'Payments')
        results, unmatched, issues, accepted = reconcile(invoices, payments)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    total_invoiced = sum((parse_amount(v, 'Invoice') for v in invoices['amount']), Decimal('0'))
    all_payment_rows = sum((parse_amount(v, 'Payment') for v in payments['amount']), Decimal('0'))
    outstanding = Decimal(str(results['remaining'].sum())).quantize(CENT)
    overpaid = Decimal(str(results['overpayment'].sum())).quantize(CENT)
    st.success('Reconciliation completed successfully.')
    st.subheader('Overview')
    a, b, c, d = st.columns(4)
    a.metric('Total invoiced', euro(total_invoiced))
    b.metric('Accepted payments', euro(accepted))
    c.metric('Outstanding', euro(outstanding))
    d.metric('Overpayments', euro(overpaid))
    st.caption(f'{len(invoices)} invoices · {len(payments)} payment rows · '
               f'{len(unmatched)} unmatched · {len(issues)} duplicate/conflict rows · '
               f'{euro(all_payment_rows)} in raw payment rows')

    st.subheader('Invoice status breakdown')
    status_counts = results['status'].value_counts().reindex(STATUS_ORDER, fill_value=0)
    st.bar_chart(status_counts.rename_axis('Status').rename('Invoices'))

    with st.expander('View uploaded data'):
        st.markdown('**Invoices**')
        st.dataframe(invoices, hide_index=True, use_container_width=True)
        st.markdown('**Payments**')
        st.dataframe(payments, hide_index=True, use_container_width=True)

    st.subheader('Reconciliation results')
    col1, col2 = st.columns([1, 2])
    with col1:
        statuses = st.multiselect('Filter by status', STATUS_ORDER)
    with col2:
        query = st.text_input('Search invoice or customer')
    filtered = results.copy()
    if statuses:
        filtered = filtered[filtered['status'].isin(statuses)]
    if query.strip():
        term = query.strip()
        filtered = filtered[
            filtered['invoice_id'].str.contains(term, case=False, regex=False) |
            filtered['customer'].str.contains(term, case=False, regex=False)
        ]
    st.caption(f'Showing {len(filtered)} of {len(results)} invoices')
    st.dataframe(filtered, hide_index=True, use_container_width=True)
    st.download_button('Download reconciliation CSV', csv_data(results),
                       'reconciliation_results.csv', 'text/csv')

    left, right = st.columns(2)
    with left:
        st.subheader('Unmatched payments')
        if unmatched.empty:
            st.success('No unmatched payments.')
        else:
            st.dataframe(unmatched, hide_index=True, use_container_width=True)
        st.download_button('Download unmatched CSV', csv_data(unmatched), 'unmatched_payments.csv', 'text/csv')
    with right:
        st.subheader('Duplicates / conflicts')
        if issues.empty:
            st.success('No duplicate or conflicting payments.')
        else:
            st.dataframe(issues, hide_index=True, use_container_width=True)
        st.download_button('Download issues CSV', csv_data(issues), 'duplicates_conflicts.csv', 'text/csv')

    st.subheader('Export report')
    try:
        workbook = excel_data({
            'Reconciliation': results, 'Unmatched': unmatched, 'Duplicates Conflicts': issues,
            'Invoices': invoices, 'Payments': payments,
        })
        st.download_button('📥 Download complete Excel report', workbook,
                           'reconciliation_report.xlsx',
                           'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', type='primary')
    except ImportError:
        st.warning('Excel export needs openpyxl: python -m pip install openpyxl')
    st.caption('Reconciliation rules: identical duplicate payment IDs count once; conflicting IDs are excluded pending review. Amounts are assumed to be in one currency (EUR).')


if __name__ == '__main__':
    main()
