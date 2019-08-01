# hcan-pytorch
hierarchical convolutional attention networks for text classification


```
from models import Proto_CNN as HCAN

model = HCAN(input_dim, hidden_dim, kernel_dim,
             sent_maxlen, dropout_rate, num_emb, pretrained_weight)
```
if you don't have ```pretrained_weight```, you should modify the Class ```Proto``` to ```pretrained_weight``` optional.

```
logtis = model(x, None)
```
second variable is length of ```x```, ```l```. However, since this variable is no longer needed, insert ```None```.   
```x```is index of words.
