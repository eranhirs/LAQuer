# LAQuer: Localized Attribution Queries in Content-grounded Generation

Repository for the paper `LAQuer: Localized Attribution Queries in Content-grounded Generation`, accepted to ACL 2025 main conference.


### What is LAQuer?

When reading LLM generated outputs, users can request attribution for specific pieces of information by highlighting them. We call this task LAQuer.

![LAQuer](imgs/LAQuer.png)

### Why localized attribution?

Our motivation: users should not have to read more than necessary!

For example, underlined green is the lengthy attribution for the entire sentence, but the user is only interested in the information highlighted yellow.

![comparison](imgs/comparison.png)


### Framework for evaluating LAQuer methods

Most existing methods already generate coarse sentence-level attribution (Stage 1). Accordingly, we propose a modeling framework that utilizes existing attribution and extends it to LAQuer (Stage 2).

![framework](imgs/framework.png)


# Code for evaluating LAQuer methods

Our evaluation setup includes the following steps:
1. Synthesis of user highlights from unseen outputs. This step is important to support novel generation methods.
2. Generating attribution to user highlights.
3. Evaluation of generated attribution.

To run LAQuer, use the `scripts/run_all.py` file. Running LAQuer assumes existing outputs in a file called `results.json`, as the examples provided in the `results` directory.



# Citation

```
@inproceedings{hirsch-etal-2025-laquer,
    title = "LAQuer: Localized Attribution Queries in Content-grounded Generation",
    author = "Hirsch, Eran  and
      Slobodkin, Aviv  and
      Wan, David  and
      Stengel-Eskin, Elias  and
      Bansal, Mohit  and
      Dagan, Ido",
    booktitle = "Proceedings of the 63nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics"
}
```