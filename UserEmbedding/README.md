## Model Training
You may choose several arguments to adjust hyper-parameters such as epoch, batch_size, and lambda. You can see the hyper-parameters list in main.py. Here, we just emphasize how to combine **TRAP** easily. 
```
$ python main.py --using_trap [1(use), 0 (not use)]
```
You can also select 3 datasets (*Movielens-1M, Yelp, and VideoGame*) and 4 baseline models (*AutoRec, CDAE, MultiVAE, and JCA*) in main.py.
