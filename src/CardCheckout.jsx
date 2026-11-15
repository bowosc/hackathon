import { ArrowRight, CreditCard } from 'lucide-react';
import { money } from './purchases.js';

export default function CardCheckout({ quote, loading, onComplete }) {
  // Demo fields stay in the DOM only. No card values are serialized, stored, or sent to the API.
  return <section className="card-entry-panel" aria-labelledby="card-details-title">
    <div className="card-entry-heading"><h2 id="card-details-title">Payment details</h2><span><CreditCard size={16} />Card</span></div>
    <form className="card-entry-form" autoComplete="off" onSubmit={event => { event.preventDefault(); onComplete(); }}>
      <label>Name on card<input placeholder="Alex Morgan" autoComplete="off" required /></label>
      <label>Card number<div className="card-number-field"><input placeholder="4242 4242 4242 4242" inputMode="numeric" autoComplete="off" maxLength={23} pattern="[0-9 ]{12,23}" required onChange={event => { event.target.value = event.target.value.replace(/\D/g, '').slice(0, 19).replace(/(.{4})/g, '$1 ').trim(); }} /><CreditCard size={19} /></div></label>
      <div className="card-entry-row">
        <label>Expiration date<input placeholder="MM/YY" inputMode="numeric" autoComplete="off" maxLength={5} pattern="(0[1-9]|1[0-2])/[0-9]{2}" required onChange={event => { const digits = event.target.value.replace(/\D/g, '').slice(0, 4); event.target.value = digits.length > 2 ? `${digits.slice(0, 2)}/${digits.slice(2)}` : digits; }} /></label>
        <label>Security code<input type="password" placeholder="CVC" inputMode="numeric" autoComplete="off" minLength={3} maxLength={4} pattern="[0-9]{3,4}" required /></label>
      </div>
      <button className="primary-button" type="submit" disabled={loading}>{loading ? 'Preparing your dataset…' : `Pay ${money(quote.amount_cents, quote.currency)}`}<ArrowRight size={17} /></button>
      <button className="demo-skip-button" type="button" disabled={loading} onClick={onComplete}>Demo: skip payment</button>
      <p className="card-entry-demo">Demo checkout · No charge</p>
    </form>
  </section>;
}
