### Model
This is a very simple model that performs basic logistic regression. It is expected to be trainable to about 91% accuracy on MNIST.

Code from [MnistTestBase.java:293](../../../../../../../src/test/java/com/simiacryptus/mindseye/opt/MnistTestBase.java#L293) executed in 0.00 seconds: 
```java
    PipelineNetwork network = new PipelineNetwork();
    network.add(new BiasLayer(28, 28, 1));
    network.add(new FullyConnectedLayer(new int[]{28, 28, 1}, new int[]{10})
      .setWeights(() -> 0.001 * (Math.random() - 0.45)));
    network.add(new SoftmaxActivationLayer());
    return network;
```

Returns: 

```
    PipelineNetwork/5d2f317b-9de0-40ef-aa3d-c4c047ee09a5
```



### Training
Code from [QuadraticLineSearchTest.java:43](../../../../../../../src/test/java/com/simiacryptus/mindseye/opt/line/QuadraticLineSearchTest.java#L43) executed in 180.44 seconds: 
```java
    SimpleLossNetwork supervisedNetwork = new SimpleLossNetwork(network, new EntropyLossLayer());
    Trainable trainable = new SampledArrayTrainable(trainingData, supervisedNetwork, 1000);
    return new IterativeTrainer(trainable)
      .setMonitor(monitor)
      .setOrientation(new GradientDescent())
      .setLineSearchFactory((String name) -> new QuadraticSearch())
      .setTimeout(3, TimeUnit.MINUTES)
      .setMaxIterations(500)
      .run();
```
Logging: 
```
    Constructing line search parameters: GD
    F(0.0) = LineSearchPoint{point=PointSample{avg=2.4860843634590073}, derivative=-494182.65544153017}
    New Minimum: 2.4860843634590073 > 2.486059593524799
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=2.486059593524799}, derivative=-494169.56219214323}, delta = -2.4769934208190847E-5
    New Minimum: 2.486059593524799 > 2.4859109876597714
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=2.4859109876597714}, derivative=-494091.00056139124}, delta = -1.7337579923593083E-4
    New Minimum: 2.4859109876597714 > 2.484871406218926
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=2.484871406218926}, derivative=-493540.96686013346}, delta = -0.0012129572400811917
    New Minimum: 2.484871406218926 > 2.4776266854842426
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=2.4776266854842426}, derivative=-489685.79480917274}, delta = -0.00845767797476471
    New Minimum: 2.4776266854842426 > 2.4285076514588764
    F(2.4010000000000004E-7) = LineS
```
...[skipping 264311 bytes](etc/155.txt)...
```
    394
    F(3.5649744793616492E-6) = LineSearchPoint{point=PointSample{avg=0.3697742646211326}, derivative=-3512.49574324996}, delta = -0.00832463659996524
    F(2.4954821355531544E-5) = LineSearchPoint{point=PointSample{avg=0.4030905099497809}, derivative=9749.738998234709}, delta = 0.024991608728683057
    F(1.9196016427331956E-6) = LineSearchPoint{point=PointSample{avg=0.3731120437047526}, derivative=-4555.643006389462}, delta = -0.0049868575163452555
    F(1.343721149913237E-5) = LineSearchPoint{point=PointSample{avg=0.36754459954512897}, derivative=2607.237443250926}, delta = -0.01055430167596888
    0.36754459954512897 <= 0.37809890122109785
    New Minimum: 0.36584334100888327 > 0.36479631139887064
    F(9.262572429005947E-6) = LineSearchPoint{point=PointSample{avg=0.36479631139887064}, derivative=40.23584738265424}, delta = -0.013302589822227207
    Right bracket at 9.262572429005947E-6
    Converged to right
    Iteration 215 complete. Error: 0.36479631139887064 Total: 59841585385229.3600; Orientation: 0.0003; Line Search: 0.8515
    
```

Returns: 

```
    0.36479631139887064
```



Code from [MnistTestBase.java:139](../../../../../../../src/test/java/com/simiacryptus/mindseye/opt/MnistTestBase.java#L139) executed in 0.00 seconds: 
```java
    PlotCanvas plot = ScatterPlot.plot(history.stream().map(step -> new double[]{step.iteration, Math.log10(step.point.getMean())}).toArray(i -> new double[i][]));
    plot.setTitle("Convergence Plot");
    plot.setAxisLabels("Iteration", "log10(Fitness)");
    plot.setSize(600, 400);
    return plot;
```

Returns: 

![Result](etc/test.697.png)



Saved model as [model0.json](etc/model0.json)

### Metrics
Code from [MnistTestBase.java:152](../../../../../../../src/test/java/com/simiacryptus/mindseye/opt/MnistTestBase.java#L152) executed in 0.05 seconds: 
```java
    try {
      ByteArrayOutputStream out = new ByteArrayOutputStream();
      JsonUtil.writeJson(out, monitoringRoot.getMetrics());
      return out.toString();
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
```

Returns: 

```
    [ "java.util.HashMap", {
      "BiasLayer/3a4f3911-35e0-428d-8680-682d4b8b1336" : [ "java.util.HashMap", {
        "avgMsPerItem" : 0.016575353662547015,
        "medianMsPerItem" : "NaN",
        "avgMsPerItem_Backward" : 2.1909232648038654E-5,
        "totalItems" : 1861000,
        "backpropStatistics" : [ "java.util.HashMap", {
          "meanExponent" : -5.002957941664053,
          "tp50" : -0.00148936379390155,
          "negative" : 192154,
          "min" : -3.241493368288857E-4,
          "max" : 4.6542518646960397E-4,
          "tp90" : -0.0013025747261692384,
          "mean" : 3.5576421144207567E-10,
          "count" : 392000.0,
          "positive" : 199846,
          "stdDev" : 1.7971151615703884E-4,
          "tp75" : -0.0013685645279855247,
          "zeros" : 0
        } ],
        "totalBatches" : 3722,
        "weights" : [ "java.util.HashMap", {
          "tp50" : "NaN",
          "buffers" : 1,
          "max" : 2.2884186385545437E-8,
          "tp90" : "NaN",
          "count" : 784.0,
          "positive" : 391,
          "tp75" : "NaN",
          "zeros" : 0,
          "m
```
...[skipping 2808 bytes](etc/156.txt)...
```
    s" : [ "java.util.HashMap", {
          "meanExponent" : 0.14390076915413133,
          "tp50" : -1.0466942334299774,
          "negative" : 500,
          "min" : -27.86386529103693,
          "max" : 0.0,
          "tp90" : -1.0024067482869194,
          "mean" : -0.688921242919489,
          "count" : 5000.0,
          "positive" : 0,
          "stdDev" : 17.346004364089303,
          "tp75" : -1.0098956843155422,
          "zeros" : 4500
        } ],
        "totalBatches" : 3722,
        "class" : "com.simiacryptus.mindseye.layers.java.SoftmaxActivationLayer",
        "outputStatistics" : [ "java.util.HashMap", {
          "meanExponent" : -3.2931461570653853,
          "tp50" : 1.7430759848564213E-7,
          "negative" : 0,
          "min" : 5.366225150042088E-12,
          "max" : 0.9880069107608973,
          "tp90" : 9.976969831348817E-7,
          "mean" : 0.1,
          "count" : 5000.0,
          "positive" : 5000,
          "stdDev" : 0.2679806918937728,
          "tp75" : 6.671745849361641E-7,
          "zeros" : 0
        } ],
        "medianMsPerItem_Backward" : "NaN"
      } ]
    } ]
```



### Validation
If we run our model against the entire validation dataset, we get this accuracy:

Code from [MnistTestBase.java:209](../../../../../../../src/test/java/com/simiacryptus/mindseye/opt/MnistTestBase.java#L209) executed in 0.81 seconds: 
```java
    return MNIST.validationDataStream().mapToDouble(labeledObject ->
      predict(network, labeledObject)[0] == parse(labeledObject.label) ? 1 : 0)
      .average().getAsDouble() * 100;
```

Returns: 

```
    91.23
```



Let's examine some incorrectly predicted results in more detail:

Code from [MnistTestBase.java:216](../../../../../../../src/test/java/com/simiacryptus/mindseye/opt/MnistTestBase.java#L216) executed in 0.05 seconds: 
```java
    try {
      TableOutput table = new TableOutput();
      MNIST.validationDataStream().map(labeledObject -> {
        try {
          int actualCategory = parse(labeledObject.label);
          double[] predictionSignal = GpuController.call(ctx -> network.eval(ctx, labeledObject.data).getData().get(0).getData());
          int[] predictionList = IntStream.range(0, 10).mapToObj(x -> x).sorted(Comparator.comparing(i -> -predictionSignal[i])).mapToInt(x -> x).toArray();
          if (predictionList[0] == actualCategory) return null; // We will only examine mispredicted rows
          LinkedHashMap<String, Object> row = new LinkedHashMap<String, Object>();
          row.put("Image", log.image(labeledObject.data.toGrayImage(), labeledObject.label));
          row.put("Prediction", Arrays.stream(predictionList).limit(3)
            .mapToObj(i -> String.format("%d (%.1f%%)", i, 100.0 * predictionSignal[i]))
            .reduce((a, b) -> a + ", " + b).get());
          return row;
        } catch (IOException e) {
          throw new RuntimeException(e);
        }
      }).filter(x -> null != x).limit(10).forEach(table::putRow);
      return table;
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
```

Returns: 

Image | Prediction
----- | ----------
![[5]](etc/test.698.png) | 6 (96.3%), 4 (1.4%), 2 (1.3%)  
![[4]](etc/test.699.png) | 6 (60.6%), 0 (25.9%), 5 (5.3%) 
![[3]](etc/test.700.png) | 2 (62.7%), 3 (29.9%), 8 (5.6%) 
![[2]](etc/test.701.png) | 7 (60.6%), 2 (31.2%), 9 (5.2%) 
![[9]](etc/test.702.png) | 4 (46.8%), 9 (23.7%), 8 (16.6%)
![[7]](etc/test.703.png) | 1 (42.0%), 7 (41.5%), 9 (6.2%) 
![[7]](etc/test.704.png) | 4 (69.1%), 9 (17.5%), 7 (11.9%)
![[2]](etc/test.705.png) | 9 (62.3%), 8 (11.2%), 4 (9.8%) 
![[9]](etc/test.706.png) | 8 (37.4%), 9 (35.3%), 2 (18.3%)
![[9]](etc/test.707.png) | 4 (46.7%), 3 (27.4%), 9 (13.9%)




