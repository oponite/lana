# Merge creates a new state; update keeps both states and changes each from a snapshot.
state analyst = state(p1: 0.72, d: 0.35)
state model = state(p1: 0.38, d: 0.40)

compose analyst, model using merge -> combined
combined.p1

compose analyst, model using update -> (next_analyst, next_model)
next_analyst.p1
next_model.p1
