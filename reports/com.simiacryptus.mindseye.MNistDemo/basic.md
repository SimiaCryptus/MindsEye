First, define a model:

Code from [MNistDemo.java:126](../../src/test/java/com/simiacryptus/mindseye/MNistDemo.java#L126) executed in 0.73 seconds: 
```java
    PipelineNetwork network = new PipelineNetwork();
    network.add(new BiasLayer(28,28,1));
    network.add(new DenseSynapseLayer(new int[]{28,28,1},new int[]{10})
      .setWeights(()->0.001*(Math.random()-0.45)));
    network.add(new BiasLayer(10));
    network.add(new ReLuActivationLayer());
    network.add(new SoftmaxActivationLayer());
    return network;
```

Returns: 

```
    PipelineNetwork/cc7d6c7e-682d-4834-8c1f-074400000001
```



Code from [MNistDemo.java:111](../../src/test/java/com/simiacryptus/mindseye/MNistDemo.java#L111) executed in 3.31 seconds: 
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



Code from [MNistDemo.java:100](../../src/test/java/com/simiacryptus/mindseye/MNistDemo.java#L100) executed in 209.32 seconds: 
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
    0.2786001517531344
```



Code from [MNistDemo.java:60](../../src/test/java/com/simiacryptus/mindseye/MNistDemo.java#L60) executed in 0.76 seconds: 
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
    91.88918891889189
```



Code from [MNistDemo.java:73](../../src/test/java/com/simiacryptus/mindseye/MNistDemo.java#L73) executed in 0.90 seconds: 
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
![[5]](etc/basic.1.png)   | 6 (99.7%), 4 (0.2%), 5 (0.0%)  
![[4]](etc/basic.2.png)   | 6 (63.8%), 0 (31.6%), 4 (1.3%) 
![[6]](etc/basic.3.png)   | 7 (31.7%), 6 (21.6%), 2 (13.1%)
![[2]](etc/basic.4.png)   | 7 (48.8%), 2 (40.2%), 9 (3.4%) 
![[9]](etc/basic.5.png)   | 4 (49.6%), 9 (27.6%), 8 (7.6%) 
![[7]](etc/basic.6.png)   | 4 (82.9%), 7 (11.1%), 9 (5.4%) 
![[2]](etc/basic.7.png)   | 9 (65.7%), 4 (11.2%), 8 (5.2%) 
![[9]](etc/basic.8.png)   | 4 (42.4%), 3 (34.5%), 9 (11.7%)
![[3]](etc/basic.9.png)   | 8 (45.7%), 3 (23.8%), 5 (18.7%)
![[6]](etc/basic.10.png)  | 5 (56.6%), 6 (39.3%), 8 (2.3%) 
![[8]](etc/basic.11.png)  | 7 (67.1%), 8 (18.6%), 9 (11.8%)
![[9]](etc/basic.12.png)  | 8 (50.7%), 3 (26.0%), 5 (15.2%)
![[3]](etc/basic.13.png)  | 5 (34.6%), 6 (32.5%), 3 (27.8%)
![[4]](etc/basic.14.png)  | 6 (38.7%), 2 (25.9%), 8 (8.3%) 
![[6]](etc/basic.15.png)  | 0 (98.0%), 6 (1.8%), 8 (0.0%)  
![[8]](etc/basic.16.png)  | 4 (42.7%), 5 (25.8%), 8 (23.2%)
![[4]](etc/basic.17.png)  | 6 (44.1%), 1 (21.2%), 4 (6.0%) 
![[2]](etc/basic.18.png)  | 3 (75.2%), 2 (22.6%), 0 (1.1%) 
![[9]](etc/basic.19.png)  | 7 (54.8%), 1 (39.7%), 8 (1.6%) 
![[2]](etc/basic.20.png)  | 7 (86.8%), 3 (8.2%), 8 (1.8%)  
![[5]](etc/basic.21.png)  | 3 (95.8%), 5 (1.7%), 6 (0.8%)  
![[6]](etc/basic.22.png)  | 4 (43.5%), 6 (31.1%), 7 (8.6%) 
![[5]](etc/basic.23.png)  | 0 (84.5%), 3 (10.8%), 8 (2.3%) 
![[9]](etc/basic.24.png)  | 4 (84.1%), 9 (12.5%), 8 (2.4%) 
![[2]](etc/basic.25.png)  | 7 (61.3%), 3 (25.0%), 2 (7.7%) 
![[9]](etc/basic.26.png)  | 4 (38.4%), 9 (30.1%), 7 (20.9%)
![[8]](etc/basic.27.png)  | 5 (51.8%), 8 (35.6%), 2 (9.7%) 
![[5]](etc/basic.28.png)  | 3 (53.5%), 5 (46.5%), 7 (0.0%) 
![[8]](etc/basic.29.png)  | 7 (78.9%), 8 (14.9%), 3 (3.1%) 
![[2]](etc/basic.30.png)  | 8 (69.3%), 2 (29.9%), 3 (0.3%) 
![[6]](etc/basic.31.png)  | 0 (98.1%), 7 (0.5%), 6 (0.4%)  
![[9]](etc/basic.32.png)  | 8 (55.5%), 5 (24.1%), 3 (13.9%)
![[3]](etc/basic.33.png)  | 5 (77.5%), 3 (22.3%), 8 (0.1%) 
![[7]](etc/basic.34.png)  | 2 (53.8%), 9 (32.9%), 7 (9.5%) 
![[5]](etc/basic.35.png)  | 8 (50.1%), 5 (18.8%), 2 (16.0%)
![[9]](etc/basic.36.png)  | 3 (55.5%), 8 (28.8%), 5 (7.3%) 
![[8]](etc/basic.37.png)  | 3 (42.8%), 2 (18.9%), 8 (15.2%)
![[5]](etc/basic.38.png)  | 3 (81.6%), 5 (7.0%), 7 (6.9%)  
![[3]](etc/basic.39.png)  | 5 (89.0%), 8 (5.7%), 3 (3.9%)  
![[4]](etc/basic.40.png)  | 1 (45.3%), 8 (27.2%), 4 (11.8%)
![[9]](etc/basic.41.png)  | 4 (68.0%), 9 (26.5%), 8 (2.8%) 
![[3]](etc/basic.42.png)  | 6 (76.3%), 3 (15.6%), 2 (3.4%) 
![[8]](etc/basic.43.png)  | 3 (69.1%), 8 (11.2%), 4 (9.4%) 
![[7]](etc/basic.44.png)  | 1 (82.1%), 3 (8.6%), 9 (3.7%)  
![[4]](etc/basic.45.png)  | 9 (63.7%), 4 (28.4%), 7 (7.1%) 
![[3]](etc/basic.46.png)  | 5 (40.8%), 3 (31.4%), 4 (24.3%)
![[3]](etc/basic.47.png)  | 2 (49.7%), 8 (33.6%), 3 (8.5%) 
![[8]](etc/basic.48.png)  | 3 (85.4%), 8 (7.0%), 2 (4.6%)  
![[2]](etc/basic.49.png)  | 8 (66.0%), 2 (21.6%), 3 (8.8%) 
![[1]](etc/basic.50.png)  | 8 (50.0%), 1 (22.5%), 3 (9.8%) 
![[9]](etc/basic.51.png)  | 4 (62.9%), 9 (23.1%), 8 (5.8%) 
![[2]](etc/basic.52.png)  | 6 (83.5%), 5 (6.4%), 2 (4.8%)  
![[7]](etc/basic.53.png)  | 4 (43.5%), 0 (28.5%), 7 (20.2%)
![[2]](etc/basic.54.png)  | 8 (34.7%), 9 (32.0%), 7 (29.5%)
![[7]](etc/basic.55.png)  | 3 (81.2%), 2 (14.6%), 1 (1.5%) 
![[8]](etc/basic.56.png)  | 4 (99.7%), 3 (0.2%), 6 (0.1%)  
![[4]](etc/basic.57.png)  | 9 (79.6%), 4 (17.9%), 8 (1.2%) 
![[0]](etc/basic.58.png)  | 6 (88.5%), 7 (2.2%), 0 (1.2%)  
![[5]](etc/basic.59.png)  | 2 (82.6%), 8 (15.4%), 5 (0.9%) 
![[2]](etc/basic.60.png)  | 8 (57.7%), 2 (40.1%), 3 (1.9%) 
![[2]](etc/basic.61.png)  | 8 (69.6%), 2 (16.9%), 3 (13.4%)
![[4]](etc/basic.62.png)  | 9 (82.5%), 4 (15.1%), 7 (1.9%) 
![[2]](etc/basic.63.png)  | 8 (50.5%), 2 (39.7%), 3 (7.2%) 
![[4]](etc/basic.64.png)  | 9 (42.3%), 4 (40.2%), 8 (7.3%) 
![[5]](etc/basic.65.png)  | 9 (51.2%), 8 (14.6%), 5 (11.7%)
![[8]](etc/basic.66.png)  | 3 (96.1%), 5 (2.2%), 8 (1.3%)  
![[8]](etc/basic.67.png)  | 7 (54.9%), 8 (20.6%), 5 (6.0%) 
![[5]](etc/basic.68.png)  | 3 (60.9%), 5 (32.5%), 8 (5.6%) 
![[9]](etc/basic.69.png)  | 4 (43.3%), 9 (29.7%), 8 (13.2%)
![[8]](etc/basic.70.png)  | 2 (52.9%), 8 (37.4%), 6 (8.7%) 
![[4]](etc/basic.71.png)  | 9 (48.3%), 4 (39.6%), 3 (8.9%) 
![[7]](etc/basic.72.png)  | 2 (51.5%), 8 (32.5%), 7 (7.6%) 
![[2]](etc/basic.73.png)  | 7 (72.8%), 2 (15.1%), 8 (6.1%) 
![[3]](etc/basic.74.png)  | 5 (69.9%), 3 (28.5%), 8 (1.5%) 
![[2]](etc/basic.75.png)  | 5 (38.5%), 8 (33.2%), 0 (18.3%)
![[8]](etc/basic.76.png)  | 9 (47.5%), 4 (26.7%), 8 (24.7%)
![[7]](etc/basic.77.png)  | 2 (79.9%), 9 (8.1%), 8 (4.5%)  
![[5]](etc/basic.78.png)  | 4 (52.3%), 5 (44.9%), 8 (1.7%) 
![[1]](etc/basic.79.png)  | 6 (63.1%), 3 (17.9%), 2 (9.6%) 
![[6]](etc/basic.80.png)  | 0 (96.5%), 3 (1.0%), 6 (0.7%)  
![[2]](etc/basic.81.png)  | 3 (75.6%), 2 (22.7%), 0 (0.6%) 
![[3]](etc/basic.82.png)  | 8 (44.4%), 2 (33.0%), 3 (21.7%)
![[9]](etc/basic.83.png)  | 4 (34.6%), 9 (31.8%), 7 (21.5%)
![[9]](etc/basic.84.png)  | 7 (53.8%), 9 (37.5%), 4 (7.6%) 
![[7]](etc/basic.85.png)  | 9 (39.5%), 7 (34.3%), 4 (11.0%)
![[6]](etc/basic.86.png)  | 0 (41.5%), 5 (30.3%), 8 (6.5%) 
![[5]](etc/basic.87.png)  | 8 (77.6%), 5 (9.6%), 2 (9.5%)  
![[8]](etc/basic.88.png)  | 3 (70.9%), 8 (26.4%), 1 (0.9%) 
![[7]](etc/basic.89.png)  | 9 (57.8%), 8 (12.2%), 1 (10.5%)
![[6]](etc/basic.90.png)  | 8 (61.8%), 6 (14.3%), 2 (13.6%)
![[2]](etc/basic.91.png)  | 6 (50.1%), 2 (27.0%), 3 (14.2%)
![[3]](etc/basic.92.png)  | 9 (41.3%), 3 (25.3%), 7 (19.0%)
![[8]](etc/basic.93.png)  | 4 (72.9%), 8 (16.4%), 5 (3.2%) 
![[5]](etc/basic.94.png)  | 8 (81.5%), 5 (17.8%), 3 (0.2%) 
![[5]](etc/basic.95.png)  | 3 (75.8%), 5 (13.4%), 8 (8.4%) 
![[8]](etc/basic.96.png)  | 2 (54.3%), 3 (38.4%), 8 (4.6%) 
![[9]](etc/basic.97.png)  | 3 (94.0%), 8 (3.4%), 5 (1.1%)  
![[4]](etc/basic.98.png)  | 6 (98.4%), 4 (1.3%), 0 (0.0%)  
![[3]](etc/basic.99.png)  | 8 (66.7%), 3 (18.5%), 2 (13.9%)
![[7]](etc/basic.100.png) | 2 (59.7%), 3 (10.4%), 9 (9.6%) 




