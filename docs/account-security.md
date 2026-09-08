# Account security and data ownership

CreatorCut's local product surface uses server-authenticated creator accounts instead of trusting a
creator ID stored by the browser. Registration creates one `creator_profiles` record and one
credential record in a single SQLite transaction. Every video, clip, decision, custom interval,
analytics import, performance report, post pack, feedback export, and generated media download is
resolved against the creator ID in the current server session.

## Credential and session controls

- Passwords use a unique 16-byte salt and PBKDF2-HMAC-SHA256 with 600,000 iterations. Plaintext
  passwords are never stored. Verification uses a constant-time digest comparison and performs
  equivalent password work for unknown email addresses.
- Email identifiers are normalized and uniquely constrained by SQLite.
- Browser sessions use a 256-bit opaque random token. Only its SHA-256 digest is stored. Cookies
  are `HttpOnly`, `SameSite=Lax`, scoped to `/`, and expire after seven days.
- Every state-changing product request also requires a separate session-bound CSRF token.
- Repeated failed logins are bounded per source-address and email pair for fifteen minutes.
- Generated clip downloads use separate random, creator-bound tokens that expire after 24 hours.
- Responses include a restrictive Content Security Policy, clickjacking protection, MIME-sniffing
  protection, and a no-referrer policy.

The password work factor follows the OWASP Password Storage Cheat Sheet's current
PBKDF2-HMAC-SHA256 recommendation and Python's guidance to use a slow, salted password KDF:

- https://cheatsheetseries.owasp.org/cheatsheets/Password_Storage_Cheat_Sheet.html
- https://docs.python.org/3/library/hashlib.html#hashlib.pbkdf2_hmac

## Administrator boundary

The ML observatory requires both the explicit `--enable-admin-dashboard` runtime switch and an
authenticated account whose persisted role is `administrator`. The first registration on a new
local database receives that role only when the local administrator mode was deliberately enabled.
Additional administrators are provisioned through `creatorcut-account`; passwords are prompted
interactively rather than accepted as command arguments.

Detailed model reports, cross-creator learning status, data-flow lineage, cohort diagnostics, and
adjustment distributions use the same administrator boundary. Hiding controls in the browser is
not treated as authorization; the backing endpoints independently verify the administrator role.

## Product telemetry and audience analytics

Explicit clip selections, rejections, custom intervals, and timestamp corrections are core
editorial telemetry and automatically inform creator-specific and creator-balanced community
editing models. There is no editorial-contribution toggle.

Cross-creator use of uploaded audience analytics is off by default. The creator can enable one
account-level setting, and every change is recorded in an append-only consent audit table with a
policy version and timestamp. Disabling it excludes that account from subsequent shared audience
calculations. Private creator-level analysis of the creator's own uploads remains available.

## Local-to-public boundary

This implementation is intentionally a loopback-only product server. It demonstrates credential
storage, sessions, CSRF protection, resource authorization, and administrator isolation without
claiming to be an internet-facing identity service. A public deployment should terminate managed
HTTPS, use a maintained identity provider with account recovery and MFA, keep structured records in
a managed database, keep media in private object storage, and preserve the same server-side
ownership checks.
