# NimbusCart Escalation Policy (Internal Support Guidance)

**Document ID:** POL-ESC-001
**Last updated:** 15 April 2026
**Owner:** Customer Experience Team

This document tells the support assistant when a conversation must be handed to a
human specialist. It is guidance for the assistant, not a customer-facing policy.

## Always escalate

The assistant must set `requires_human = true` and hand over when any of the
following is true.

### Money movement disputes

- The customer disputes a refund decision or a refund amount
- The customer says they were charged twice or charged the wrong amount
- The customer mentions a chargeback, their bank, or legal action
- A refund has not arrived more than 12 business days after the return arrived
- A request to refund to a different card or account

Automated support cannot issue, reverse, or promise a payment. It may explain the
policy, then escalate.

### Physical harm, safety, and damage

- A product caused injury, fire, smoke, burning smell, or an electric shock
- A parcel arrived damaged or the item was damaged on arrival
- A shipment is lost or has been missing beyond the delay thresholds

### Strong customer emotion

- The customer is angry, insulting, or threatening
- The customer says they have contacted support repeatedly without resolution
- The customer threatens to post publicly, contact a regulator, or sue
- The customer expresses distress about the impact of the problem

Never argue with an upset customer. Acknowledge the problem in one sentence,
avoid defending the company, and escalate.

### Identity, privacy, and fraud

- Any request to change account ownership or the registered email
- Any suspected account takeover or unauthorised purchase
- Any order that was cancelled by our fraud screening system
- Any request that would require sharing another person's data

### Missing information the assistant cannot obtain

- The customer refuses to provide, or does not know, their order ID
- The order ID given does not exist in our system after one clarification
- The request depends on information that is not in the knowledge base and not in
  the order database

## Do not escalate

Handle these normally, without a human:

- Any policy question answerable from the knowledge base
- Any order, product, or refund-estimate lookup answerable from the database
- Simple follow-up questions in an ongoing conversation
- Requests to explain a policy the customer disagrees with but has not disputed

## How to escalate

1. Answer whatever part of the question you legitimately can from the knowledge
   base or a tool.
2. State plainly that a human specialist will take over.
3. Give the reason in one short sentence.
4. Set `requires_human = true` in the structured response.
5. Never promise a specific outcome, refund amount, or compensation on the
   specialist's behalf.

## Response time commitments

| Case type | Human response time |
|---|---|
| Payment dispute or duplicate charge | Within 4 hours |
| Damaged on arrival | Within 4 hours |
| Product safety incident | Within 1 hour |
| Refund dispute | Within 1 business day |
| Everything else | Within 1 business day |

Human specialists are available Monday to Friday 9:00 AM to 8:00 PM Eastern Time
and Saturday 10:00 AM to 4:00 PM Eastern Time. Cases raised outside those hours
are queued for the next working period.
