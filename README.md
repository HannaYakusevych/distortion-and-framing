# distortion-and-framing

# Evaluation

## Original paper

### Causality
| model | cau | corr | uncl | mF1 |
|--|--|--|--|--|
| RoBERTa | .56 | .62 | .59 | .57+-.01 |
| SciBERT | .58 | .57 | .60 | .54+-.04 |

### Certainty

| model | c | s_c | unc | mF1 |
|--|--|--|--|--|
| RoBERTa | .67 | .48 | .51 | .59+-.04 |
| SciBERT | .70 | .50 | .60 | .53+-.02 |

### Generalization

| model | same | r_f | p_f | mF1 |
|--|--|--|--|--|
| RoBERTa | .32 | .69 | .42 | .40+-.06 |
| SciBERT | .32 | .72 | .49 | .47+-.04 |

### Sensationalism

| model | r |
|--|--|
| RoBERTa | .61+-.02 |
| SciBERT | .57+-.03 |

## Baselines

### Causality
| model | no_rel | cau | corr | no_ment |
|--|--|--|--|--|
| RoBERTa | .40 | .56 | .61 | .54 |

### Certainty

| model | c | s_c | s_unc | unc |
|--|--|--|--|--|
| RoBERTa | .66 | .48 | .37 | .07 |

### Generalization

| model | p_f | same | r_f |
|--|--|--|--|
| RoBERTa | .44 | .45 | .74 |

## Multitask models

### 2 classification tasks v1

#### Causality
| model | no_rel | cau | corr | no_ment |
|--|--|--|--|--|
| loss = loss_caus + loss_cert | .40 | .54 | .52 | .52 |
| loss = loss_caus | .38 | .54 | .56 | .38 |
| loss = loss_cert | .13 | .25 | .42 | .34 |
| loss = 0.7 * loss_caus + 0.3 * loss_cert | .57 | .56 | .53 | .53 |

#### Certainty
| model | c | s_c | s_unc | unc |
|--|--|--|--|--|
| loss = loss_caus + loss_cert | .67 | .50 | .36 | .19 |
| loss = loss_caus | .09 | .50 | .10 | .10 |
| loss = loss_cert | .66 | .42 | .41 | .08 |
| loss = 0.7 * loss_caus + 0.3 * loss_cert | .67 | .25 | .40 | .22 |
