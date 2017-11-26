### Model
This is a very simple model that performs basic logistic regression. It is expected to be trainable to about 91% accuracy on MNIST.

Code from [MnistTestBase.java:272](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/mnist/MnistTestBase.java#L272) executed in 0.00 seconds: 
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
    PipelineNetwork/e1035fb9-1fe3-4846-a360-62290000000a
```



### Training
Code from [QuadraticLineSearchTest.java:43](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/opt/line/QuadraticLineSearchTest.java#L43) executed in 180.61 seconds: 
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
    Returning cached value; 2 buffers unchanged since 0.0 => 2.5629102336749554
    F(0.0) = LineSearchPoint{point=PointSample{avg=2.5629102336749554}, derivative=-542689.5964178878}
    New Minimum: 2.5629102336749554 > 2.5628833736797487
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=2.5628833736797487}, derivative=-542671.6536499622}, delta = -2.6859995206685028E-5
    New Minimum: 2.5628833736797487 > 2.562722232607343
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=2.562722232607343}, derivative=-542563.999534474}, delta = -1.8800106761229785E-4
    New Minimum: 2.562722232607343 > 2.5615951521453453
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=2.5615951521453453}, derivative=-541810.5407307594}, delta = -0.0013150815296101293
    New Minimum: 2.5615951521453453 > 2.5537499995038644
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=2.5537499995038644}, derivative=-536542.3629144948}, delta = -0.00916023417109102
    New Minimum: 2.5537499995038644 > 2.5009966104168764
    F(2.4010000000000004E-7) = LineSearchPoint{point=PointSample{avg=2.5009966104168764}, derivative=-500012.49900408677}, delta = -0.06191362325807903
    New Minimum: 2.5009966104168764 > 2.2300007993046105
    F(1.6807000000000003E-6) = LineSearchPoint{point=PointSample{avg=2.2300007993046105}, derivative=-274728.3633517809}, delta = -0.3329094343703449
    F(1.1764900000000001E-5) = LineSearchPoint{point=PointSample{avg=2.610973688340347}, derivative=239238.90911751371}, delta = 0.048063454665391525
    F(1.1764900000000001E-7) = LineSearchPoint{point=PointSample{avg=2.5319302792353837}, derivative=-521669.43440666323}, delta = -0.03097995443957169
    F(8.235430000000001E-7) = LineSearchPoint{point=PointSample{avg=2.3714148966637767}, derivative=-401277.4598236619}, delta = -0.1914953370111787
    New Minimum: 2.2300007993046105 > 2.085899859269844
    F(5.7648010000000005E-6) = LineSearchPoint{point=PointSample{avg=2.085899859269844}, derivative=66630.64559801338}, delta = -0.4770103744
```
...[skipping 171691 bytes](etc/1.txt)...
```
    g=0.33706372459297734}, derivative=62.82171019442689}
    F(1.567703286376458E-5) = LineSearchPoint{point=PointSample{avg=0.33705358633728166}, derivative=2.502684815380464}, delta = -0.014475108918167945
    Right bracket at 1.567703286376458E-5
    Converged to right
    Iteration 89 complete. Error: 0.33705358633728166 Total: 180280417712032.7500; Orientation: 0.0006; Line Search: 1.9004
    Returning cached value; 2 buffers unchanged since 0.0 => 0.3519751103879528
    F(0.0) = LineSearchPoint{point=PointSample{avg=0.3519751103879528}, derivative=-11106.508306683816}
    New Minimum: 0.3519751103879528 > 0.3438104941329295
    F(1.567703286376458E-5) = LineSearchPoint{point=PointSample{avg=0.3438104941329295}, derivative=8624.40711440811}, delta = -0.008164616255023294
    0.3438104941329295 <= 0.3519751103879528
    New Minimum: 0.3438104941329295 > 0.3286200213509458
    Returning cached value; 2 buffers unchanged since 0.0 => 0.3286200213509458
    isLeft=false; isBracketed=true; leftPoint=LineSearchPoint{point=PointSample{avg=0.3519751103879528}, derivative=-11106.508306683816}; rightPoint=LineSearchPoint{point=PointSample{avg=0.3438104941329295}, derivative=8624.40711440811}
    F(8.824582742847755E-6) = LineSearchPoint{point=PointSample{avg=0.3286200213509458}, derivative=364.29390906893565}, delta = -0.023355089037007004
    Right bracket at 8.824582742847755E-6
    New Minimum: 0.3286200213509458 > 0.3286014559195639
    Returning cached value; 2 buffers unchanged since 0.0 => 0.3286014559195639
    isLeft=false; isBracketed=true; leftPoint=LineSearchPoint{point=PointSample{avg=0.3519751103879528}, derivative=-11106.508306683816}; rightPoint=LineSearchPoint{point=PointSample{avg=0.3286200213509458}, derivative=364.29390906893565}
    F(8.544328434314782E-6) = LineSearchPoint{point=PointSample{avg=0.3286014559195639}, derivative=24.61128410775488}, delta = -0.023373654468388905
    Right bracket at 8.544328434314782E-6
    Converged to right
    Iteration 90 complete. Error: 0.3286014559195639 Total: 180281520836590.6000; Orientation: 0.0005; Line Search: 0.8451
    
```

Returns: 

```
    0.3286014559195639
```



Code from [MnistTestBase.java:131](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/mnist/MnistTestBase.java#L131) executed in 0.03 seconds: 
```java
    PlotCanvas plot = ScatterPlot.plot(history.stream().map(step -> new double[]{step.iteration, Math.log10(step.point.getMean())}).toArray(i -> new double[i][]));
    plot.setTitle("Convergence Plot");
    plot.setAxisLabels("Iteration", "log10(Fitness)");
    plot.setSize(600, 400);
    return plot;
```

Returns: 

![Result](etc/test.1.png)



Saved model as [model0.json](etc/model0.json)

### Metrics
Code from [MnistTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/mnist/MnistTestBase.java#L144) executed in 0.08 seconds: 
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
      "SoftmaxActivationLayer/e1035fb9-1fe3-4846-a360-62290000000d" : [ "java.util.HashMap", {
        "avgMsPerItem" : 0.0037904962826747733,
        "medianMsPerItem" : "NaN",
        "avgMsPerItem_Backward" : 2.5476063556231023E-6,
        "totalItems" : 658000,
        "backpropStatistics" : [ "java.util.HashMap", {
          "meanExponent" : -2.5561529606499085,
          "tp50" : -0.0021643722580351756,
          "negative" : 500,
          "min" : -0.0991447428050771,
          "max" : 0.0,
          "tp90" : -0.002011163905992002,
          "mean" : -6.291922974107975E-4,
          "count" : 5000.0,
          "positive" : 0,
          "stdDev" : 0.011187648139418803,
          "tp75" : -0.002049312662962529,
          "zeros" : 4500
        } ],
        "totalBatches" : 1316,
        "class" : "com.simiacryptus.mindseye.layers.java.SoftmaxActivationLayer",
        "outputStatistics" : [ "java.util.HashMap", {
          "meanExponent" : -2.8104293498321478,
          "tp50" : 2.0733421167991396E-6,
          "negative" : 0,
          "min" : 1.582478451477227E-8,
          "max" : 0.9945549225322479,
          "tp90" : 8.77479240571079E-6,
          "mean" : 0.1,
          "count" : 5000.0,
          "positive" : 5000,
          "stdDev" : 0.2588204221407428,
          "tp75" : 5.491892813322415E-6,
          "zeros" : 0
        } ],
        "medianMsPerItem_Backward" : "NaN"
      } ],
      "BiasLayer/e1035fb9-1fe3-4846-a360-62290000000b" : [ "java.util.HashMap", {
        "avgMsPerItem" : 0.021412316933130702,
        "medianMsPerItem" : "NaN",
        "avgMsPerItem_Backward" : 2.9643503167173275E-5,
        "totalItems" : 658000,
        "backpropStatistics" : [ "java.util.HashMap", {
          "meanExponent" : -7.507683773202984,
          "tp50" : -2.303438274064275E-6,
          "negative" : 192955,
          "min" : -1.325538373847737E-6,
          "max" : 1.2828111577185632E-6,
          "tp90" : -1.989747900767966E-6,
          "mean" : 1.4058285046965556E-9,
          "count" : 392000.0,
          "positive" : 199045,
          "stdDev" : 3.035137896463675E-7,
          "tp75" : -2.089785915130262E-6,
          "zeros" : 0
        } ],
```
...[skipping 778 bytes](etc/2.txt)...
```
     "tp90" : -1.147808749741543E-8,
          "mean" : 33.68192346892257,
          "count" : 392000.0,
          "positive" : 217645,
          "stdDev" : 78.93033637805053,
          "tp75" : -1.147808749741543E-8,
          "zeros" : 0
        } ],
        "medianMsPerItem_Backward" : "NaN"
      } ],
      "FullyConnectedLayer/e1035fb9-1fe3-4846-a360-62290000000c" : [ "java.util.HashMap", {
        "avgMsPerItem" : 0.014735135518237076,
        "medianMsPerItem" : "NaN",
        "avgMsPerItem_Backward" : 7.726057027355619E-5,
        "totalItems" : 658000,
        "backpropStatistics" : [ "java.util.HashMap", {
          "meanExponent" : -5.614359609006677,
          "tp50" : -1.5188908231931708E-4,
          "negative" : 500,
          "min" : -0.0019596549460230667,
          "max" : 0.0017837983342344087,
          "tp90" : -1.1101935509821852E-5,
          "mean" : 7.279248947470747E-22,
          "count" : 5000.0,
          "positive" : 4500,
          "stdDev" : 2.4065028020965495E-4,
          "tp75" : -4.812605109387386E-5,
          "zeros" : 0
        } ],
        "totalBatches" : 1316,
        "weights" : [ "java.util.HashMap", {
          "tp50" : "NaN",
          "buffers" : 1,
          "max" : 0.0021565397082931284,
          "tp90" : "NaN",
          "count" : 7840.0,
          "positive" : 4306,
          "tp75" : "NaN",
          "zeros" : 0,
          "meanExponent" : -3.661790455097161,
          "negative" : 3534,
          "min" : -0.0016427015162876891,
          "mean" : 5.056478537486408E-5,
          "stdDev" : 4.041666854613679E-4
        } ],
        "class" : "com.simiacryptus.mindseye.layers.java.FullyConnectedLayer",
        "outputStatistics" : [ "java.util.HashMap", {
          "meanExponent" : 0.2439878591545723,
          "tp50" : -3.5680005887012465,
          "negative" : 1722,
          "min" : -5.241564398700268,
          "max" : 10.648346713393785,
          "tp90" : -2.7200303388454805,
          "mean" : 1.4232031968101213,
          "count" : 5000.0,
          "positive" : 3278,
          "stdDev" : 3.344983894936401,
          "tp75" : -2.9698546540838078,
          "zeros" : 0
        } ],
        "medianMsPerItem_Backward" : "NaN"
      } ]
    } ]
```



### Validation
If we run our model against the entire validation dataset, we get this accuracy:

Code from [MnistTestBase.java:201](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/mnist/MnistTestBase.java#L201) executed in 1.11 seconds: 
```java
    return MNIST.validationDataStream().mapToDouble(labeledObject ->
      predict(network, labeledObject)[0] == parse(labeledObject.label) ? 1 : 0)
      .average().getAsDouble() * 100;
```

Returns: 

```
    89.64999999999999
```



Let's examine some incorrectly predicted results in more detail:

Code from [MnistTestBase.java:208](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/mnist/MnistTestBase.java#L208) executed in 0.10 seconds: 
```java
    try {
      TableOutput table = new TableOutput();
      MNIST.validationDataStream().map(labeledObject -> {
        try {
          int actualCategory = parse(labeledObject.label);
          double[] predictionSignal = CudaExecutionContext.gpuContexts.run(ctx -> network.eval(ctx, labeledObject.data).getData().get(0).getData());
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
![[5]](etc/test.2.png)  | 6 (90.8%), 2 (4.6%), 0 (1.3%)  
![[4]](etc/test.3.png)  | 6 (50.3%), 0 (38.2%), 4 (4.0%) 
![[1]](etc/test.4.png)  | 3 (53.0%), 1 (19.7%), 8 (7.4%) 
![[3]](etc/test.5.png)  | 2 (58.9%), 3 (30.5%), 8 (5.8%) 
![[2]](etc/test.6.png)  | 7 (78.3%), 2 (12.2%), 9 (7.0%) 
![[7]](etc/test.7.png)  | 9 (51.0%), 7 (45.0%), 4 (2.1%) 
![[7]](etc/test.8.png)  | 1 (47.3%), 7 (23.7%), 9 (11.7%)
![[7]](etc/test.9.png)  | 9 (48.9%), 4 (36.2%), 7 (9.2%) 
![[2]](etc/test.10.png) | 9 (42.0%), 8 (25.7%), 6 (10.2%)
![[3]](etc/test.11.png) | 8 (30.4%), 3 (27.1%), 5 (14.1%)




