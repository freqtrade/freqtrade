export default function Terms() {
  return (
    <div className="min-h-screen">
      <nav className="border-b border-dark-700 px-6 py-4">
        <a href="/" className="flex items-center gap-2">
          <span className="text-2xl">📈</span>
          <span className="text-xl font-bold text-white">TrendRider</span>
        </a>
      </nav>

      <div className="max-w-3xl mx-auto px-6 py-8 prose prose-invert">
        <h1 className="text-3xl font-bold text-white">Terms & Conditions</h1>
        <p className="text-sm text-gray-500">Last updated: March 2026</p>

        <div className="space-y-6 text-gray-300 text-sm leading-relaxed">
          <section>
            <h2 className="text-xl font-semibold text-white">1. Acceptance of Terms</h2>
            <p>
              By connecting your wallet and using TrendRider (&quot;the Platform&quot;), you agree
              to these Terms & Conditions. If you do not agree, do not use the Platform.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-semibold text-white">2. Service Description</h2>
            <p>
              TrendRider provides automated cryptocurrency trading software that executes
              trades on your MEXC exchange account using a predefined strategy. The Platform
              does not hold, custody, or have withdrawal access to your funds.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-semibold text-white">3. Risk Disclosure</h2>
            <div className="bg-red-500/10 border border-red-500/20 rounded-lg p-4">
              <p className="font-semibold text-red-400">
                CRYPTOCURRENCY TRADING INVOLVES SUBSTANTIAL RISK OF LOSS.
              </p>
              <ul className="list-disc list-inside mt-2 space-y-1">
                <li>Past performance does not guarantee future results</li>
                <li>You may lose some or all of your invested capital</li>
                <li>The trading strategy is experimental and under active development</li>
                <li>Software bugs, exchange outages, and network issues may cause unexpected losses</li>
                <li>Market conditions can change rapidly and unpredictably</li>
              </ul>
            </div>
          </section>

          <section>
            <h2 className="text-xl font-semibold text-white">4. No Financial Advice</h2>
            <p>
              TrendRider does not provide financial, investment, or trading advice.
              The Platform is a software tool only. You are solely responsible for
              your trading decisions and the consequences thereof.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-semibold text-white">5. API Key Security</h2>
            <ul className="list-disc list-inside space-y-1">
              <li>Your MEXC API keys are encrypted with AES-256-GCM before storage</li>
              <li>Keys are decrypted only in memory when operating your bot</li>
              <li>You must only enable Spot Trading permissions — never Withdrawal</li>
              <li>You are responsible for the security of your exchange account</li>
              <li>TrendRider is not liable for unauthorized access to your account</li>
            </ul>
          </section>

          <section>
            <h2 className="text-xl font-semibold text-white">6. Access Control</h2>
            <p>
              Access to the Platform is by invitation only. Admin may revoke access
              at any time without notice. Whitelisting does not constitute a guarantee
              of service availability or performance.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-semibold text-white">7. Limitation of Liability</h2>
            <p>
              TO THE MAXIMUM EXTENT PERMITTED BY LAW, TRENDRIDER AND ITS OPERATORS
              SHALL NOT BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
              CONSEQUENTIAL, OR EXEMPLARY DAMAGES, INCLUDING BUT NOT LIMITED TO
              TRADING LOSSES, LOST PROFITS, OR LOSS OF DATA.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-semibold text-white">8. No Warranty</h2>
            <p>
              THE PLATFORM IS PROVIDED &quot;AS IS&quot; WITHOUT WARRANTY OF ANY KIND,
              EXPRESS OR IMPLIED. WE DO NOT WARRANT THAT THE SERVICE WILL BE
              UNINTERRUPTED, ERROR-FREE, OR PROFITABLE.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-semibold text-white">9. Modifications</h2>
            <p>
              We reserve the right to modify the trading strategy, platform features,
              or these terms at any time. Continued use after changes constitutes
              acceptance of the modified terms.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-semibold text-white">10. USE AT YOUR OWN RISK</h2>
            <div className="bg-yellow-500/10 border border-yellow-500/20 rounded-lg p-4 font-semibold text-yellow-200">
              By using TrendRider, you acknowledge that you have read, understood,
              and agree to these terms. You accept full responsibility for any and
              all outcomes resulting from the use of this Platform.
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}
