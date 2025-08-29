# ID90 OTA Growth Model (Weekly, Take-Focused)

This model simulates **weekly growth, engagement, bookings, and take (revenue share)** across ID90’s main lines of business (Flights, Hotels, Cars, Cruises). It’s designed to be configurable through `config.yaml`.

## Core Flow

1. **Active Users**

   * Start with an initial active base (past 12mo).
   * Each week add **new signups** (from per-group weekly baselines + growth rates).
   * Subtract churn (monthly churn converted to weekly, split by group share).

2. **Engaged Users**

   * A portion of active users are engaged (log in, browse, etc).
   * Engaged decays weekly (`weekly_engaged_churn`) but grows as new signups convert (`acct_create_rate × new_user_retention_90d`).

3. **Bookers**

   * Each week, a share of engaged users become bookers (`booker_share_weekly`).

4. **Bookings Allocation**

   * For each group (partners vs non-partners):

     * Apply **product mix weights** (historical purchase combinations).
     * Apply **conversion funnel strength** (either staged or a single full-funnel rate per LOB/group).
     * Apply **per-LOB purchase frequencies** (avg bookings per booker per year, group-specific).
     * Combine these factors into normalized weights.
     * Multiply by bookers × group’s weekly purchase frequency → **bookings per LOB**.
   * Flight bookings for non-partners are blocked unless `commercial_flights_enabled: true`.

5. **Take Calculation**

   * Flights: `$ per segment × avg segments per booking`.
   * Hotels, Cars, Cruises: `$ per booking`.
   * Multiply bookings × take → weekly take per LOB.
   * Sum → **total take**.

## Outputs

The simulation produces a CSV with weekly rows:

* Active, engaged, bookers (by group)
* Bookings and take (by LOB)
* Totals across all LOBs


