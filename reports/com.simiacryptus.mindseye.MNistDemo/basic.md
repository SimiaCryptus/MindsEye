First, define a model:

This is a very simple model that performs basic logistic regression. It is expected to be trainable to about 91% accuracy on MNIST.

Code from [MNistDemo.java:137](../.././src/test/java/com/simiacryptus/mindseye/MNistDemo.java#L137) executed in 0.78 seconds: 
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
    PipelineNetwork/fcd30a0f-ce26-4a55-aa93-165a00000001
```



We use the standard MNIST dataset, made available by a helper function. In order to use data, we convert it into data tensors; helper functions are defined to work with images.

Code from [MNistDemo.java:120](../.././src/test/java/com/simiacryptus/mindseye/MNistDemo.java#L120) executed in 3.72 seconds: 
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

Code from [MNistDemo.java:106](../.././src/test/java/com/simiacryptus/mindseye/MNistDemo.java#L106) executed in 143.37 seconds: 
```java
    SimpleLossNetwork supervisedNetwork = new SimpleLossNetwork(network, new EntropyLossLayer());
    StochasticArrayTrainable trainable = new StochasticArrayTrainable(trainingData, supervisedNetwork, 1000);
    return new IterativeTrainer(trainable)
        .setTimeout(5, TimeUnit.MINUTES)
        .setMaxIterations(500)
        .run();
```
Logging: 
```
    Constructing line search parameters: GD
    Constructing line search parameters: LBFGS
    
```

Returns: 

```
    0.32795182725991334
```



If we test our model against the entire validation dataset, we get this accuracy:

Code from [MNistDemo.java:61](../.././src/test/java/com/simiacryptus/mindseye/MNistDemo.java#L61) executed in 0.91 seconds: 
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
    91.36741022306693
```



Let's examine some incorrectly predicted results in more detail:

Code from [MNistDemo.java:75](../.././src/test/java/com/simiacryptus/mindseye/MNistDemo.java#L75) executed in 0.96 seconds: 
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
![[5]](etc/basic.1.png)   | 6 (99.3%), 4 (0.4%), 5 (0.1%)  
![[4]](etc/basic.2.png)   | 6 (56.7%), 0 (31.4%), 5 (7.1%) 
![[3]](etc/basic.3.png)   | 2 (60.8%), 3 (34.0%), 8 (4.4%) 
![[6]](etc/basic.4.png)   | 7 (28.5%), 2 (22.8%), 6 (20.4%)
![[2]](etc/basic.5.png)   | 7 (47.0%), 2 (36.6%), 9 (10.9%)
![[2]](etc/basic.6.png)   | 8 (48.5%), 2 (36.0%), 9 (8.5%) 
![[7]](etc/basic.7.png)   | 4 (83.8%), 9 (11.8%), 7 (3.9%) 
![[2]](etc/basic.8.png)   | 9 (88.9%), 4 (5.1%), 8 (2.9%)  
![[5]](etc/basic.9.png)   | 8 (41.9%), 5 (41.5%), 3 (12.0%)
![[9]](etc/basic.10.png)  | 3 (37.3%), 9 (28.6%), 4 (27.4%)
![[3]](etc/basic.11.png)  | 8 (47.6%), 3 (27.6%), 5 (15.2%)
![[5]](etc/basic.12.png)  | 7 (42.5%), 5 (23.2%), 0 (15.2%)
![[6]](etc/basic.13.png)  | 5 (65.3%), 6 (24.1%), 8 (5.7%) 
![[8]](etc/basic.14.png)  | 7 (42.9%), 9 (35.5%), 8 (20.0%)
![[9]](etc/basic.15.png)  | 8 (58.1%), 3 (19.1%), 5 (10.8%)
![[3]](etc/basic.16.png)  | 5 (41.3%), 6 (30.6%), 3 (22.9%)
![[4]](etc/basic.17.png)  | 2 (26.6%), 6 (21.4%), 4 (20.9%)
![[6]](etc/basic.18.png)  | 0 (99.5%), 6 (0.5%), 8 (0.0%)  
![[8]](etc/basic.19.png)  | 4 (64.1%), 5 (14.7%), 8 (12.2%)
![[4]](etc/basic.20.png)  | 6 (37.8%), 1 (29.8%), 4 (10.4%)
![[7]](etc/basic.21.png)  | 9 (75.8%), 7 (15.0%), 8 (4.8%) 
![[2]](etc/basic.22.png)  | 3 (79.0%), 2 (15.2%), 0 (4.8%) 
![[9]](etc/basic.23.png)  | 7 (54.9%), 1 (34.1%), 9 (5.8%) 
![[2]](etc/basic.24.png)  | 7 (87.1%), 3 (5.1%), 9 (3.3%)  
![[5]](etc/basic.25.png)  | 3 (96.4%), 5 (1.5%), 2 (0.7%)  
![[6]](etc/basic.26.png)  | 4 (50.3%), 6 (33.3%), 7 (6.1%) 
![[5]](etc/basic.27.png)  | 0 (93.9%), 3 (2.8%), 8 (2.7%)  
![[7]](etc/basic.28.png)  | 9 (51.6%), 7 (42.2%), 8 (1.9%) 
![[9]](etc/basic.29.png)  | 4 (59.2%), 9 (39.4%), 8 (1.2%) 
![[2]](etc/basic.30.png)  | 7 (72.0%), 3 (13.4%), 8 (7.4%) 
![[5]](etc/basic.31.png)  | 3 (65.9%), 5 (33.8%), 8 (0.1%) 
![[8]](etc/basic.32.png)  | 7 (58.5%), 8 (30.4%), 9 (7.7%) 
![[2]](etc/basic.33.png)  | 8 (86.1%), 2 (13.3%), 3 (0.4%) 
![[6]](etc/basic.34.png)  | 0 (99.6%), 6 (0.2%), 7 (0.2%)  
![[9]](etc/basic.35.png)  | 8 (54.7%), 9 (29.3%), 5 (8.0%) 
![[3]](etc/basic.36.png)  | 5 (67.2%), 3 (32.0%), 8 (0.5%) 
![[7]](etc/basic.37.png)  | 9 (71.2%), 2 (14.1%), 7 (10.5%)
![[5]](etc/basic.38.png)  | 8 (69.5%), 2 (10.6%), 5 (10.3%)
![[9]](etc/basic.39.png)  | 3 (46.0%), 8 (35.0%), 9 (7.3%) 
![[8]](etc/basic.40.png)  | 0 (40.7%), 3 (25.6%), 8 (18.9%)
![[5]](etc/basic.41.png)  | 3 (83.9%), 7 (7.0%), 5 (3.4%)  
![[3]](etc/basic.42.png)  | 5 (81.2%), 8 (12.9%), 3 (4.4%) 
![[4]](etc/basic.43.png)  | 8 (37.7%), 1 (33.6%), 4 (14.9%)
![[3]](etc/basic.44.png)  | 6 (57.2%), 3 (29.4%), 2 (6.5%) 
![[2]](etc/basic.45.png)  | 1 (83.3%), 2 (14.0%), 3 (2.5%) 
![[8]](etc/basic.46.png)  | 3 (42.8%), 4 (32.1%), 8 (14.5%)
![[7]](etc/basic.47.png)  | 9 (62.5%), 7 (32.3%), 3 (2.3%) 
![[7]](etc/basic.48.png)  | 1 (73.6%), 9 (13.3%), 3 (9.1%) 
![[4]](etc/basic.49.png)  | 9 (84.6%), 4 (14.4%), 7 (0.7%) 
![[3]](etc/basic.50.png)  | 5 (37.6%), 4 (31.2%), 3 (26.6%)
![[3]](etc/basic.51.png)  | 8 (56.4%), 2 (23.6%), 9 (15.5%)
![[2]](etc/basic.52.png)  | 8 (58.6%), 2 (36.9%), 9 (3.6%) 
![[8]](etc/basic.53.png)  | 3 (88.0%), 8 (7.5%), 2 (2.6%)  
![[7]](etc/basic.54.png)  | 9 (47.7%), 7 (33.5%), 2 (10.2%)
![[2]](etc/basic.55.png)  | 8 (70.3%), 2 (18.9%), 3 (8.5%) 
![[1]](etc/basic.56.png)  | 8 (47.5%), 1 (28.9%), 3 (9.0%) 
![[2]](etc/basic.57.png)  | 8 (37.6%), 1 (28.8%), 2 (28.0%)
![[9]](etc/basic.58.png)  | 4 (59.1%), 9 (34.2%), 8 (3.8%) 
![[3]](etc/basic.59.png)  | 9 (64.5%), 3 (26.6%), 8 (4.1%) 
![[2]](etc/basic.60.png)  | 6 (84.1%), 2 (5.5%), 5 (4.8%)  
![[7]](etc/basic.61.png)  | 4 (44.6%), 0 (26.7%), 9 (14.6%)
![[2]](etc/basic.62.png)  | 9 (74.8%), 8 (17.0%), 7 (6.2%) 
![[7]](etc/basic.63.png)  | 3 (82.5%), 2 (11.2%), 1 (3.2%) 
![[7]](etc/basic.64.png)  | 9 (51.7%), 4 (20.6%), 7 (20.4%)
![[8]](etc/basic.65.png)  | 4 (99.4%), 3 (0.3%), 9 (0.2%)  
![[4]](etc/basic.66.png)  | 9 (93.0%), 4 (6.5%), 8 (0.3%)  
![[0]](etc/basic.67.png)  | 6 (89.6%), 9 (2.7%), 0 (2.7%)  
![[5]](etc/basic.68.png)  | 2 (74.3%), 8 (23.4%), 5 (0.9%) 
![[2]](etc/basic.69.png)  | 8 (81.1%), 2 (17.2%), 3 (1.4%) 
![[2]](etc/basic.70.png)  | 8 (77.3%), 2 (15.9%), 3 (6.7%) 
![[4]](etc/basic.71.png)  | 9 (94.1%), 4 (5.5%), 7 (0.3%)  
![[2]](etc/basic.72.png)  | 8 (58.0%), 2 (30.2%), 3 (7.6%) 
![[4]](etc/basic.73.png)  | 9 (59.3%), 4 (30.0%), 8 (7.1%) 
![[5]](etc/basic.74.png)  | 9 (61.2%), 8 (18.9%), 2 (6.3%) 
![[8]](etc/basic.75.png)  | 3 (93.7%), 5 (3.0%), 8 (2.9%)  
![[8]](etc/basic.76.png)  | 7 (49.7%), 8 (28.0%), 4 (8.4%) 
![[5]](etc/basic.77.png)  | 3 (69.3%), 5 (14.6%), 8 (14.1%)
![[8]](etc/basic.78.png)  | 2 (53.2%), 8 (43.8%), 6 (2.3%) 
![[4]](etc/basic.79.png)  | 9 (63.7%), 4 (28.3%), 3 (6.4%) 
![[7]](etc/basic.80.png)  | 8 (44.0%), 2 (34.4%), 7 (8.7%) 
![[2]](etc/basic.81.png)  | 7 (75.1%), 8 (11.4%), 2 (6.9%) 
![[7]](etc/basic.82.png)  | 9 (42.1%), 7 (31.5%), 2 (10.7%)
![[3]](etc/basic.83.png)  | 5 (56.2%), 3 (38.0%), 8 (5.5%) 
![[2]](etc/basic.84.png)  | 0 (45.2%), 8 (33.4%), 5 (17.4%)
![[8]](etc/basic.85.png)  | 9 (85.6%), 4 (9.1%), 8 (5.2%)  
![[7]](etc/basic.86.png)  | 2 (58.7%), 9 (28.5%), 8 (6.4%) 
![[5]](etc/basic.87.png)  | 4 (61.5%), 5 (34.4%), 8 (2.5%) 
![[1]](etc/basic.88.png)  | 6 (45.7%), 3 (27.8%), 2 (13.0%)
![[4]](etc/basic.89.png)  | 9 (41.7%), 4 (19.9%), 8 (17.1%)
![[6]](etc/basic.90.png)  | 0 (99.0%), 3 (0.4%), 5 (0.3%)  
![[2]](etc/basic.91.png)  | 3 (73.8%), 2 (24.3%), 0 (1.2%) 
![[3]](etc/basic.92.png)  | 8 (49.0%), 3 (28.4%), 2 (21.3%)
![[5]](etc/basic.93.png)  | 3 (55.5%), 5 (37.3%), 8 (7.1%) 
![[7]](etc/basic.94.png)  | 9 (74.2%), 7 (12.5%), 4 (7.3%) 
![[6]](etc/basic.95.png)  | 5 (65.1%), 0 (19.7%), 8 (9.8%) 
![[6]](etc/basic.96.png)  | 2 (60.3%), 6 (39.2%), 4 (0.5%) 
![[5]](etc/basic.97.png)  | 8 (83.2%), 5 (9.4%), 2 (3.7%)  
![[8]](etc/basic.98.png)  | 3 (59.9%), 8 (35.7%), 1 (3.1%) 
![[7]](etc/basic.99.png)  | 9 (72.6%), 8 (10.8%), 1 (6.1%) 
![[6]](etc/basic.100.png) | 8 (62.9%), 2 (18.8%), 6 (8.7%) 




