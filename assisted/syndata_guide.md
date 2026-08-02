Synthetic Q&A scaling guide
Short answer

No — 3 Q&A pairs from 60 chunks is only about 180 examples, which is enough for a first proof of concept but not the same thing as the 1,000 to 5,000 example target we discussed for a stronger chatbot run.
What you are missing

You are not missing anything conceptually. You are just still in the bootstrap stage.

The earlier larger target assumed a more developed supervised dataset with substantial variation. Right now you have:

    a chunked source corpus,

    a local model that can help generate training data,

    a working fine-tuning path.

That means the missing piece is simply dataset scale plus quality control.
The math

If you do:

    60 chunks × 3 Q&A pairs = 180 examples

    200 chunks × 3 Q&A pairs = 600 examples

    400 chunks × 3 Q&A pairs = 1,200 examples

    500 chunks × 4 Q&A pairs = 2,000 examples

That is how you grow into the 1k to 5k range.
The real bottleneck

The limiting factor is usually not source text quantity. It is whether the generated Q&A pairs are:

    grounded,

    non-repetitive,

    varied in question style,

    useful for chat behavior.

If you generate 3,000 bad synthetic examples, you may get worse behavior than 500 decent ones.
Recommended ladder
Stage 1: proof of concept

    50 to 75 chunks

    2 to 3 Q&A pairs each

    target: 100 to 225 examples

This is enough to check whether the model learns the interaction shape at all.
Stage 2: first serious run

    200 to 400 chunks

    3 to 4 Q&A pairs each

    target: 600 to 1,600 examples

This is where the model starts feeling more like a real domain chatbot.
Stage 3: stronger version

    500 plus chunks

    3 to 6 Q&A pairs each

    target: 1,500 to 3,000 plus examples

    with filtering and some human review

That is much closer to the kind of dataset size that usually produces noticeably more reliable task behavior.
Best practice for your setup

Because you can use local models in LM Studio, the best workflow is:

    Generate synthetic Q&A from chunks.

    Score and filter it.

    Hand-review a sample.

    Train on the filtered set.

    Expand gradually.

Important idea

You do not need 5,000 examples on day one.

You need:

    enough examples to teach the pattern,

    enough quality control to avoid junk,

    enough iteration to see what the model is still failing at.

That is why a 150 to 300 example run can still be very useful even though it is below the longer-term target.