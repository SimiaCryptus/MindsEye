First, define a model:

This is a very simple model that performs basic logistic regression. It is expected to be trainable to about 91% accuracy on MNIST.

Code from [MNistDemo.java:137](../../src/test/java/com/simiacryptus/mindseye/MNistDemo.java#L137) executed in 0.95 seconds: 
```java
    PipelineNetwork network = new PipelineNetwork();
    network.add(new BiasLayer(28,28,1));
    network.add(new DenseSynapseLayer(new int[]{28,28,1},new int[]{10})
      .setWeights(()->0.001*(Math.random()-0.45)));
    network.add(new SoftmaxActivationLayer());
    return network;
```

Returns: 

```
    PipelineNetwork/b972ebcf-164e-4958-9425-0b8e00000001
```



We use the standard MNIST dataset, made available by a helper function. In order to use data, we convert it into data tensors; helper functions are defined to work with images.

Code from [MNistDemo.java:120](../../src/test/java/com/simiacryptus/mindseye/MNistDemo.java#L120) executed in 3.72 seconds: 
```java
    try {
      return MNIST.trainingDataStream().map(labeledObject -> {
        Tensor categoryTensor = new Tensor(10);
        int category = Integer.parseInt(labeledObject.label.replaceAll("[^\\d]", ""));
        categoryTensor.set(category, 1);
        return new Tensor[]{labeledObject.data, categoryTensor};
      }).toArray(i->new Tensor[i][]);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
```

Returns: 

```
    [[Lcom.simiacryptus.util.ml.Tensor;@46daef40
```



Training a model involves a few different components. First, our model is combined with a loss function. Then we take that model and combine it with our training data to define a trainable object. Finally, we use a simple iterative scheme to refine the weights of our model. The final output is the last output value of the loss function when evaluating the last batch.

Code from [MNistDemo.java:106](../../src/test/java/com/simiacryptus/mindseye/MNistDemo.java#L106) executed in 188.57 seconds: 
```java
    SimpleLossNetwork supervisedNetwork = new SimpleLossNetwork(network, new EntropyLossLayer());
    StochasticArrayTrainable trainable = new StochasticArrayTrainable(trainingData, supervisedNetwork, 1000);
    return new IterativeTrainer(trainable)
        .setTimeout(5, TimeUnit.MINUTES)
        .setMaxIterations(500)
        .run();
```

Returns: 

```
    0.30256166411319924
```



If we test our model against the entire validation dataset, we get this accuracy:

Code from [MNistDemo.java:61](../../src/test/java/com/simiacryptus/mindseye/MNistDemo.java#L61) executed in 0.71 seconds: 
```java
    try {
      return MNIST.validationDataStream().mapToDouble(labeledObject->{
        int actualCategory = Integer.parseInt(labeledObject.label.replaceAll("[^\\d]", ""));
        double[] predictionSignal = network.eval(labeledObject.data).data[0].getData();
        int[] predictionList = IntStream.range(0, 10).mapToObj(x -> x).sorted(Comparator.comparing(i -> -predictionSignal[i])).mapToInt(x -> x).toArray();
        return predictionList[0]==actualCategory?1:0;
      }).average().getAsDouble() * 100;
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
```

Returns: 

```
    92.15921592159216
```



Let's examine some incorrectly predicted results in more detail:

Code from [MNistDemo.java:75](../../src/test/java/com/simiacryptus/mindseye/MNistDemo.java#L75) executed in 0.77 seconds: 
```java
    try {
      TableOutput table = new TableOutput();
      MNIST.validationDataStream().map(labeledObject->{
        try {
          int actualCategory = Integer.parseInt(labeledObject.label.replaceAll("[^\\d]", ""));
          double[] predictionSignal = network.eval(labeledObject.data).data[0].getData();
          int[] predictionList = IntStream.range(0, 10).mapToObj(x -> x).sorted(Comparator.comparing(i -> -predictionSignal[i])).mapToInt(x -> x).toArray();
          if(predictionList[0] == actualCategory) return null; // We will only examine mispredicted rows
          LinkedHashMap<String, Object> row = new LinkedHashMap<String, Object>();
          row.put("Image", log.image(labeledObject.data.toGrayImage(),labeledObject.label));
          row.put("Prediction", Arrays.stream(predictionList).limit(3)
                                    .mapToObj(i->String.format("%d (%.1f%%)",i, 100.0*predictionSignal[i]))
                                    .reduce((a,b)->a+", "+b).get());
          return row;
        } catch (IOException e) {
          throw new RuntimeException(e);
        }
      }).filter(x->null!=x).limit(100).forEach(table::putRow);
      return table;
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
```

Returns: 

Image | Prediction
----- | ----------
![[5]](etc/basic.1.png)   | 6 (99.2%), 4 (0.3%), 2 (0.2%)  
![[4]](etc/basic.2.png)   | 6 (58.2%), 0 (28.0%), 5 (9.0%) 
![[3]](etc/basic.3.png)   | 2 (65.6%), 3 (31.4%), 8 (2.7%) 
![[6]](etc/basic.4.png)   | 2 (35.8%), 7 (29.3%), 6 (18.3%)
![[9]](etc/basic.5.png)   | 4 (48.7%), 9 (29.1%), 8 (12.6%)
![[7]](etc/basic.6.png)   | 4 (74.2%), 9 (18.4%), 7 (6.7%) 
![[2]](etc/basic.7.png)   | 9 (84.3%), 4 (6.4%), 8 (3.0%)  
![[9]](etc/basic.8.png)   | 3 (49.7%), 4 (26.0%), 9 (17.1%)
![[3]](etc/basic.9.png)   | 8 (40.3%), 3 (32.1%), 5 (15.1%)
![[5]](etc/basic.10.png)  | 7 (53.7%), 5 (25.0%), 0 (8.6%) 
![[6]](etc/basic.11.png)  | 5 (69.8%), 6 (23.8%), 8 (4.0%) 
![[8]](etc/basic.12.png)  | 7 (55.3%), 9 (25.4%), 8 (17.2%)
![[9]](etc/basic.13.png)  | 8 (49.5%), 3 (24.9%), 5 (14.1%)
![[3]](etc/basic.14.png)  | 5 (47.0%), 3 (26.1%), 6 (23.9%)
![[4]](etc/basic.15.png)  | 2 (51.3%), 6 (20.0%), 3 (12.7%)
![[6]](etc/basic.16.png)  | 0 (97.5%), 6 (2.4%), 8 (0.1%)  
![[8]](etc/basic.17.png)  | 4 (41.1%), 5 (31.4%), 8 (18.0%)
![[4]](etc/basic.18.png)  | 6 (34.3%), 1 (33.2%), 2 (7.3%) 
![[7]](etc/basic.19.png)  | 9 (66.0%), 7 (19.5%), 8 (6.8%) 
![[3]](etc/basic.20.png)  | 5 (56.8%), 3 (36.0%), 4 (4.6%) 
![[2]](etc/basic.21.png)  | 3 (50.7%), 2 (45.7%), 0 (2.6%) 
![[9]](etc/basic.22.png)  | 7 (56.1%), 1 (33.7%), 9 (5.2%) 
![[2]](etc/basic.23.png)  | 7 (77.1%), 3 (10.1%), 8 (5.1%) 
![[5]](etc/basic.24.png)  | 3 (93.4%), 5 (2.8%), 2 (2.0%)  
![[5]](etc/basic.25.png)  | 0 (78.8%), 3 (15.0%), 8 (3.5%) 
![[9]](etc/basic.26.png)  | 4 (69.2%), 9 (29.0%), 8 (1.4%) 
![[2]](etc/basic.27.png)  | 7 (49.5%), 3 (30.1%), 8 (9.8%) 
![[0]](etc/basic.28.png)  | 5 (51.4%), 0 (48.0%), 2 (0.4%) 
![[8]](etc/basic.29.png)  | 7 (50.0%), 8 (35.1%), 3 (6.7%) 
![[2]](etc/basic.30.png)  | 8 (72.0%), 2 (27.2%), 3 (0.4%) 
![[6]](etc/basic.31.png)  | 0 (99.2%), 7 (0.4%), 6 (0.3%)  
![[9]](etc/basic.32.png)  | 8 (47.2%), 5 (20.2%), 9 (18.6%)
![[3]](etc/basic.33.png)  | 5 (71.1%), 3 (28.2%), 2 (0.5%) 
![[7]](etc/basic.34.png)  | 9 (44.1%), 2 (41.0%), 7 (11.1%)
![[5]](etc/basic.35.png)  | 8 (53.7%), 2 (26.7%), 5 (10.9%)
![[9]](etc/basic.36.png)  | 3 (52.7%), 8 (29.9%), 5 (7.6%) 
![[8]](etc/basic.37.png)  | 2 (48.3%), 3 (20.4%), 8 (12.2%)
![[5]](etc/basic.38.png)  | 3 (78.9%), 5 (7.6%), 7 (5.9%)  
![[3]](etc/basic.39.png)  | 5 (86.1%), 8 (8.0%), 3 (4.6%)  
![[4]](etc/basic.40.png)  | 8 (39.7%), 1 (35.2%), 4 (10.8%)
![[3]](etc/basic.41.png)  | 6 (74.3%), 3 (14.3%), 2 (7.4%) 
![[2]](etc/basic.42.png)  | 1 (56.4%), 2 (40.3%), 3 (3.2%) 
![[8]](etc/basic.43.png)  | 3 (56.5%), 4 (14.8%), 8 (12.9%)
![[7]](etc/basic.44.png)  | 1 (68.3%), 3 (15.1%), 9 (9.4%) 
![[4]](etc/basic.45.png)  | 9 (85.4%), 4 (12.0%), 7 (2.1%) 
![[3]](etc/basic.46.png)  | 5 (51.3%), 4 (23.1%), 3 (22.3%)
![[3]](etc/basic.47.png)  | 2 (49.1%), 8 (35.7%), 9 (7.9%) 
![[8]](etc/basic.48.png)  | 2 (41.6%), 8 (26.7%), 3 (18.9%)
![[8]](etc/basic.49.png)  | 3 (82.8%), 2 (9.1%), 8 (5.9%)  
![[2]](etc/basic.50.png)  | 8 (51.5%), 2 (35.3%), 3 (10.8%)
![[1]](etc/basic.51.png)  | 8 (50.4%), 1 (16.3%), 5 (14.0%)
![[9]](etc/basic.52.png)  | 4 (59.7%), 9 (31.6%), 8 (5.0%) 
![[3]](etc/basic.53.png)  | 9 (48.1%), 3 (39.8%), 8 (5.0%) 
![[2]](etc/basic.54.png)  | 6 (68.6%), 2 (14.1%), 5 (8.0%) 
![[7]](etc/basic.55.png)  | 4 (36.0%), 0 (29.1%), 9 (17.1%)
![[2]](etc/basic.56.png)  | 9 (59.9%), 8 (25.7%), 7 (9.0%) 
![[7]](etc/basic.57.png)  | 3 (54.8%), 2 (42.1%), 1 (1.9%) 
![[7]](etc/basic.58.png)  | 9 (43.0%), 7 (24.7%), 4 (21.6%)
![[8]](etc/basic.59.png)  | 4 (99.4%), 3 (0.4%), 9 (0.2%)  
![[4]](etc/basic.60.png)  | 9 (90.8%), 4 (7.8%), 8 (1.0%)  
![[0]](etc/basic.61.png)  | 6 (87.7%), 7 (4.3%), 9 (3.0%)  
![[5]](etc/basic.62.png)  | 2 (84.1%), 8 (14.5%), 5 (0.7%) 
![[2]](etc/basic.63.png)  | 8 (65.4%), 2 (33.2%), 3 (1.3%) 
![[2]](etc/basic.64.png)  | 8 (60.8%), 2 (31.8%), 3 (7.3%) 
![[4]](etc/basic.65.png)  | 9 (94.1%), 4 (5.0%), 7 (0.5%)  
![[4]](etc/basic.66.png)  | 9 (53.4%), 4 (31.9%), 8 (9.1%) 
![[5]](etc/basic.67.png)  | 9 (49.5%), 8 (23.6%), 2 (14.0%)
![[8]](etc/basic.68.png)  | 3 (93.6%), 5 (5.0%), 8 (1.2%)  
![[8]](etc/basic.69.png)  | 7 (42.1%), 8 (30.1%), 5 (11.9%)
![[5]](etc/basic.70.png)  | 3 (63.3%), 5 (25.3%), 8 (8.2%) 
![[8]](etc/basic.71.png)  | 2 (78.9%), 8 (17.8%), 6 (3.0%) 
![[4]](etc/basic.72.png)  | 9 (55.1%), 4 (24.5%), 3 (16.9%)
![[3]](etc/basic.73.png)  | 5 (52.9%), 3 (41.0%), 8 (3.7%) 
![[7]](etc/basic.74.png)  | 2 (57.9%), 8 (29.9%), 3 (3.9%) 
![[2]](etc/basic.75.png)  | 7 (66.0%), 2 (16.6%), 8 (9.6%) 
![[3]](etc/basic.76.png)  | 5 (68.8%), 3 (29.1%), 8 (2.0%) 
![[2]](etc/basic.77.png)  | 8 (33.7%), 5 (27.4%), 0 (26.4%)
![[8]](etc/basic.78.png)  | 9 (75.7%), 8 (12.0%), 4 (11.6%)
![[7]](etc/basic.79.png)  | 2 (88.3%), 9 (6.6%), 8 (2.6%)  
![[1]](etc/basic.80.png)  | 6 (47.5%), 2 (29.2%), 3 (16.2%)
![[4]](etc/basic.81.png)  | 9 (26.0%), 4 (21.5%), 3 (17.3%)
![[6]](etc/basic.82.png)  | 0 (95.8%), 5 (1.6%), 3 (1.3%)  
![[3]](etc/basic.83.png)  | 2 (46.9%), 8 (27.5%), 3 (24.4%)
![[7]](etc/basic.84.png)  | 9 (52.2%), 7 (24.9%), 4 (8.7%) 
![[6]](etc/basic.85.png)  | 5 (55.2%), 0 (29.9%), 8 (7.9%) 
![[6]](etc/basic.86.png)  | 2 (54.1%), 6 (45.7%), 4 (0.2%) 
![[5]](etc/basic.87.png)  | 8 (71.4%), 2 (13.0%), 5 (11.9%)
![[8]](etc/basic.88.png)  | 3 (69.9%), 8 (27.0%), 2 (1.7%) 
![[7]](etc/basic.89.png)  | 9 (59.9%), 8 (14.0%), 1 (9.2%) 
![[6]](etc/basic.90.png)  | 8 (45.3%), 2 (34.9%), 6 (15.6%)
![[2]](etc/basic.91.png)  | 6 (47.1%), 2 (28.6%), 3 (17.1%)
![[3]](etc/basic.92.png)  | 9 (47.7%), 3 (25.8%), 7 (16.5%)
![[8]](etc/basic.93.png)  | 4 (55.3%), 8 (24.6%), 6 (6.3%) 
![[5]](etc/basic.94.png)  | 8 (57.4%), 5 (42.0%), 9 (0.3%) 
![[5]](etc/basic.95.png)  | 3 (76.7%), 5 (11.7%), 8 (7.5%) 
![[7]](etc/basic.96.png)  | 9 (57.1%), 7 (28.7%), 0 (12.5%)
![[8]](etc/basic.97.png)  | 2 (83.0%), 3 (13.7%), 8 (1.5%) 
![[9]](etc/basic.98.png)  | 3 (90.9%), 8 (4.8%), 5 (2.5%)  
![[4]](etc/basic.99.png)  | 6 (97.4%), 4 (2.0%), 2 (0.3%)  
![[3]](etc/basic.100.png) | 8 (65.3%), 2 (17.0%), 3 (16.9%)




