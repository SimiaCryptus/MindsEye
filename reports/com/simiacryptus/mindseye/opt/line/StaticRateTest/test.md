### Model
This is a very simple model that performs basic logistic regression. It is expected to be trainable to about 91% accuracy on MNIST.

Code from [MnistTestBase.java:295](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/mnist/MnistTestBase.java#L295) executed in 0.05 seconds: 
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
    PipelineNetwork/3cc8990a-29bd-4377-9ee9-863800000013
```



### Training
Code from [StaticRateTest.java:43](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/opt/line/StaticRateTest.java#L43) executed in 181.04 seconds: 
```java
    SimpleLossNetwork supervisedNetwork = new SimpleLossNetwork(network, new EntropyLossLayer());
    Trainable trainable = new SampledArrayTrainable(trainingData, supervisedNetwork, 1000);
    return new IterativeTrainer(trainable)
      .setMonitor(monitor)
      .setOrientation(new GradientDescent())
      .setLineSearchFactory((String name) -> new StaticLearningRate().setRate(0.001))
      .setTimeout(3, TimeUnit.MINUTES)
      .setMaxIterations(500)
      .run();
```
Logging: 
```
    Constructing line search parameters: GD
    Returning cached value; 2 buffers unchanged since 0.0 => 2.504815609627717
    Non-decreasing runStep. 39.239822388129156 > 5.009631219255434 at 0.001
    Non-decreasing runStep. 39.01432847124121 > 5.009631219255434 at 5.0E-4
    Non-decreasing runStep. 37.71611072668938 > 5.009631219255434 at 2.5E-4
    Non-decreasing runStep. 33.13450336957684 > 5.009631219255434 at 1.25E-4
    Non-decreasing runStep. 23.92399622765467 > 5.009631219255434 at 6.25E-5
    Non-decreasing runStep. 12.730703142405424 > 5.009631219255434 at 3.125E-5
    Non-decreasing runStep. 6.375752081374701 > 5.009631219255434 at 1.5625E-5
    New Minimum: 2.504815609627717 > 2.0414771267856793
    Iteration 1 complete. Error: 2.0414771267856793 Total: 12538853164919.9630; Orientation: 0.0003; Line Search: 1.4028
    Returning cached value; 2 buffers unchanged since 0.0 => 2.0929682076295038
    Non-decreasing runStep. 33.32255177820345 > 4.1859364152590075 at 0.001
    Non-decreasing runStep. 30.368548813064173 > 4.1859364152590075 at 5.0E-4
    Non-decreasing runStep. 25.812380213458148 > 4.1859364152590075 at 2.5E-4
    Non-decreasing runStep. 21.406715730489147 > 4.1859364152590075 at 1.25E-4
    Non-decreasing runStep. 18.766560874299206 > 4.1859364152590075 at 6.25E-5
    Non-decreasing runStep. 14.88660994828354 > 4.1859364152590075 at 3.125E-5
    Non-decreasing runStep. 9.326811307269043 > 4.1859364152590075 at 1.5625E-5
    Non-decreasing runStep. 5.142015007172894 > 4.1859364152590075 at 7.8125E-6
    New Minimum: 2.0929682076295038 > 1.7291649880338529
    Iteration 2 complete. Error: 1.7291649880338529 Total: 12540591499593.1460; Orientation: 0.0004; Line Search: 1.5619
    Returning cached value; 2 buffers unchanged since 0.0 => 1.7216582330054182
    Non-decreasing runStep. 45.40850809740177 > 3.4433164660108364 at 0.001
    Non-decreasing runStep. 45.069535645112296 > 3.4433164660108364 at 5.0E-4
    Non-decreasing runStep. 44.25425055335431 > 3.4433164660108364 at 2.5E-4
    Non-decreasing runStep. 40.34082840110179 > 3.4433164660108364 at 1.25E-4
    Non-decreasin
```
...[skipping 85140 bytes](etc/1.txt)...
```
    22548 at 5.0E-4
    Non-decreasing runStep. 20.966678439647012 > 0.6842680864322548 at 2.5E-4
    Non-decreasing runStep. 10.014288072470567 > 0.6842680864322548 at 1.25E-4
    Non-decreasing runStep. 3.6701638067674605 > 0.6842680864322548 at 6.25E-5
    Non-decreasing runStep. 1.4048738952646578 > 0.6842680864322548 at 3.125E-5
    Non-decreasing runStep. 0.7468839831992788 > 0.6842680864322548 at 1.5625E-5
    New Minimum: 0.3421340432161274 > 0.3211730215590011
    Iteration 113 complete. Error: 0.3211730215590011 Total: 12715217170543.3750; Orientation: 0.0003; Line Search: 1.4165
    Returning cached value; 2 buffers unchanged since 0.0 => 0.3143583145820418
    Non-decreasing runStep. 50.02250556442888 > 0.6287166291640836 at 0.001
    Non-decreasing runStep. 46.44237300461148 > 0.6287166291640836 at 5.0E-4
    Non-decreasing runStep. 32.70369167897446 > 0.6287166291640836 at 2.5E-4
    Non-decreasing runStep. 12.78858532026084 > 0.6287166291640836 at 1.25E-4
    Non-decreasing runStep. 3.4169262412196577 > 0.6287166291640836 at 6.25E-5
    Non-decreasing runStep. 1.0523888808649922 > 0.6287166291640836 at 3.125E-5
    Non-decreasing runStep. 0.6465693221448076 > 0.6287166291640836 at 1.5625E-5
    New Minimum: 0.3143583145820418 > 0.29631582346822627
    Iteration 114 complete. Error: 0.29631582346822627 Total: 12716686533602.7830; Orientation: 0.0003; Line Search: 1.3070
    Returning cached value; 2 buffers unchanged since 0.0 => 0.3377864573573045
    Non-decreasing runStep. 44.46942511704915 > 0.675572914714609 at 0.001
    Non-decreasing runStep. 32.49820353811554 > 0.675572914714609 at 5.0E-4
    Non-decreasing runStep. 16.587492536587117 > 0.675572914714609 at 2.5E-4
    Non-decreasing runStep. 5.738893099423934 > 0.675572914714609 at 1.25E-4
    Non-decreasing runStep. 1.8768263438995574 > 0.675572914714609 at 6.25E-5
    Non-decreasing runStep. 0.8475450821285203 > 0.675572914714609 at 3.125E-5
    New Minimum: 0.3377864573573045 > 0.3307838369514503
    Iteration 115 complete. Error: 0.3307838369514503 Total: 12718078479870.3120; Orientation: 0.0003; Line Search: 1.2319
    
```

Returns: 

```
    0.3307838369514503
```



Code from [MnistTestBase.java:141](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/mnist/MnistTestBase.java#L141) executed in 0.02 seconds: 
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
Code from [MnistTestBase.java:154](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/mnist/MnistTestBase.java#L154) executed in 0.06 seconds: 
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
      "SoftmaxActivationLayer/3cc8990a-29bd-4377-9ee9-863800000016" : [ "java.util.HashMap", {
        "avgMsPerItem" : 0.0021216905913043404,
        "medianMsPerItem" : "NaN",
        "avgMsPerItem_Backward" : 1.2687727787439647E-6,
        "totalItems" : 1035000,
        "backpropStatistics" : [ "java.util.HashMap", {
          "meanExponent" : -2.564674619903352,
          "tp50" : -0.0021035551507777896,
          "negative" : 500,
          "min" : -0.3411041833053131,
          "max" : 0.0,
          "tp90" : -0.0020053817781819656,
          "mean" : -0.001307277341230056,
          "count" : 5000.0,
          "positive" : 0,
          "stdDev" : 0.04576805097669675,
          "tp75" : -0.0020301641605051184,
          "zeros" : 4500
        } ],
        "totalBatches" : 2070,
        "class" : "com.simiacryptus.mindseye.layers.java.SoftmaxActivationLayer",
        "outputStatistics" : [ "java.util.HashMap", {
          "meanExponent" : -3.1194102248764333,
          "tp50" : 3.3030701720291265E-7,
          "negative" : 0,
          "min" : 6.621585653552045E-9,
          "max" : 0.9926109115848682,
          "tp90" : 1.8226597645232458E-6,
          "mean" : 0.1,
          "count" : 5000.0,
          "positive" : 5000,
          "stdDev" : 0.26519089854630185,
          "tp75" : 1.0604691852602119E-6,
          "zeros" : 0
        } ],
        "medianMsPerItem_Backward" : "NaN"
      } ],
      "BiasLayer/3cc8990a-29bd-4377-9ee9-863800000014" : [ "java.util.HashMap", {
        "avgMsPerItem" : 0.014239422018357473,
        "medianMsPerItem" : "NaN",
        "avgMsPerItem_Backward" : 1.9335550902415426E-5,
        "totalItems" : 1035000,
        "backpropStatistics" : [ "java.util.HashMap", {
          "meanExponent" : -7.631625123700779,
          "tp50" : -2.5439584301433848E-6,
          "negative" : 190030,
          "min" : -1.3527789008625482E-6,
          "max" : 1.1179541309650814E-6,
          "tp90" : -2.253531467286505E-6,
          "mean" : 1.7229114235443826E-9,
          "count" : 392000.0,
          "positive" : 201970,
          "stdDev" : 3.1756763715572966E-7,
          "tp75" : -2.3468573201687786E-6,
          "zeros" : 0
```
...[skipping 784 bytes](etc/2.txt)...
```
     "tp90" : -2.0153040810430302E-8,
          "mean" : 34.103183672200636,
          "count" : 392000.0,
          "positive" : 223387,
          "stdDev" : 79.36430899184126,
          "tp75" : -2.0153040810430302E-8,
          "zeros" : 0
        } ],
        "medianMsPerItem_Backward" : "NaN"
      } ],
      "FullyConnectedLayer/3cc8990a-29bd-4377-9ee9-863800000015" : [ "java.util.HashMap", {
        "avgMsPerItem" : 0.008709532178743939,
        "medianMsPerItem" : "NaN",
        "avgMsPerItem_Backward" : 5.02092852077295E-5,
        "totalItems" : 1035000,
        "backpropStatistics" : [ "java.util.HashMap", {
          "meanExponent" : -5.940146951616193,
          "tp50" : -9.845727195647802E-5,
          "negative" : 500,
          "min" : -0.0019882733774730064,
          "max" : 0.0017709169528084953,
          "tp90" : -5.367335278018314E-6,
          "mean" : -1.3237312535314428E-21,
          "count" : 5000.0,
          "positive" : 4500,
          "stdDev" : 2.3202600373596887E-4,
          "tp75" : -2.971598168456794E-5,
          "zeros" : 0
        } ],
        "totalBatches" : 2070,
        "weights" : [ "java.util.HashMap", {
          "tp50" : "NaN",
          "buffers" : 1,
          "max" : 0.0024022061201165604,
          "tp90" : "NaN",
          "count" : 7840.0,
          "positive" : 4307,
          "tp75" : "NaN",
          "zeros" : 0,
          "meanExponent" : -3.629531222062309,
          "negative" : 3533,
          "min" : -0.0020539594885861483,
          "mean" : 5.563723765608002E-5,
          "stdDev" : 4.434086239294081E-4
        } ],
        "class" : "com.simiacryptus.mindseye.layers.java.FullyConnectedLayer",
        "outputStatistics" : [ "java.util.HashMap", {
          "meanExponent" : 0.28158374184132967,
          "tp50" : -4.290419228544492,
          "negative" : 1838,
          "min" : -6.059401898122911,
          "max" : 10.31580379390442,
          "tp90" : -3.3490261385358093,
          "mean" : 1.344980878692598,
          "count" : 5000.0,
          "positive" : 3162,
          "stdDev" : 3.784478121791865,
          "tp75" : -3.678751969737622,
          "zeros" : 0
        } ],
        "medianMsPerItem_Backward" : "NaN"
      } ]
    } ]
```



### Validation
If we run our model against the entire validation dataset, we get this accuracy:

Code from [MnistTestBase.java:211](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/mnist/MnistTestBase.java#L211) executed in 0.73 seconds: 
```java
    return MNIST.validationDataStream().mapToDouble(labeledObject ->
      predict(network, labeledObject)[0] == parse(labeledObject.label) ? 1 : 0)
      .average().getAsDouble() * 100;
```

Returns: 

```
    89.8
```



Let's examine some incorrectly predicted results in more detail:

Code from [MnistTestBase.java:218](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/mnist/MnistTestBase.java#L218) executed in 0.06 seconds: 
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
![[5]](etc/test.2.png)  | 6 (91.4%), 2 (3.5%), 4 (1.9%)  
![[4]](etc/test.3.png)  | 6 (44.0%), 0 (35.5%), 5 (9.9%) 
![[1]](etc/test.4.png)  | 3 (59.4%), 1 (26.8%), 5 (5.3%) 
![[3]](etc/test.5.png)  | 2 (48.7%), 3 (46.5%), 8 (2.0%) 
![[6]](etc/test.6.png)  | 2 (44.1%), 6 (28.4%), 7 (12.4%)
![[9]](etc/test.7.png)  | 7 (56.7%), 9 (30.9%), 8 (7.1%) 
![[2]](etc/test.8.png)  | 7 (79.9%), 2 (16.5%), 9 (1.4%) 
![[3]](etc/test.9.png)  | 5 (55.4%), 3 (43.3%), 6 (0.6%) 
![[9]](etc/test.10.png) | 4 (48.2%), 9 (16.8%), 8 (13.0%)
![[7]](etc/test.11.png) | 4 (65.9%), 7 (15.3%), 9 (14.9%)




