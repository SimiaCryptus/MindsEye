### Model
This is a very simple model that performs basic logistic regression. It is expected to be trainable to about 91% accuracy on MNIST.

Code from [MnistTestBase.java:272](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/mnist/MnistTestBase.java#L272) executed in 0.34 seconds: 
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
    PipelineNetwork/e1035fb9-1fe3-4846-a360-622900000001
```



### Training
Code from [BisectionLineSearchTest.java:43](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/opt/line/BisectionLineSearchTest.java#L43) executed in 180.26 seconds: 
```java
    SimpleLossNetwork supervisedNetwork = new SimpleLossNetwork(network, new EntropyLossLayer());
    Trainable trainable = new SampledArrayTrainable(trainingData, supervisedNetwork, 1000);
    return new IterativeTrainer(trainable)
      .setMonitor(monitor)
      .setOrientation(new GradientDescent())
      .setLineSearchFactory((String name) -> new BisectionSearch())
      .setTimeout(3, TimeUnit.MINUTES)
      .setMaxIterations(500)
      .run();
```
Logging: 
```
    Found 2 devices
    Device 0 - GeForce GTX 1080 Ti
    Device 1 - GeForce GTX 1060 6GB
    Found 2 devices; using devices [0, 1]
    Constructing line search parameters: GD
    Returning cached value; 2 buffers unchanged since 0.0 => 2.3823126470559757
    F(0.0) = LineSearchPoint{point=PointSample{avg=2.3823126470559757}, derivative=-327202.2598781368}
    F(1.0)@0 = LineSearchPoint{point=PointSample{avg=14.754965275905844}, derivative=-1.1261687106732928E-14}
    Right is at most 1.0
    F(0.5)@1 = LineSearchPoint{point=PointSample{avg=14.75496527592652}, derivative=-1.4726949335418443E-9}
    Right is at most 0.5
    F(0.25)@2 = LineSearchPoint{point=PointSample{avg=14.754965427989351}, derivative=-1.0830857573283178E-5}
    Right is at most 0.25
    F(0.125)@3 = LineSearchPoint{point=PointSample{avg=14.754978231871625}, derivative=-9.167566008226447E-4}
    Right is at most 0.125
    F(0.0625)@4 = LineSearchPoint{point=PointSample{avg=14.7550792654604}, derivative=-0.007673880884417766}
    Right is at most 0.0625
    F(0.03125)@5 = LineSearchPoint{point=PointSample{avg=14.755279711297256}, derivative=0.42194483273009326}
    Right is at most 0.03125
    F(0.015625)@6 = LineSearchPoint{point=PointSample{avg=14.73925183737581}, derivative=12.460793456170926}
    Right is at most 0.015625
    F(0.0078125)@7 = LineSearchPoint{point=PointSample{avg=14.670425479966312}, derivative=20.77559738165624}
    Right is at most 0.0078125
    F(0.00390625)@8 = LineSearchPoint{point=PointSample{avg=14.616768369722134}, derivative=53.015464092186534}
    Right is at most 0.00390625
    F(0.001953125)@9 = LineSearchPoint{point=PointSample{avg=14.485059150768173}, derivative=334.5371925762646}
    Right is at most 0.001953125
    F(9.765625E-4)@10 = LineSearchPoint{point=PointSample{avg=14.162056942032962}, derivative=1379.3743675776136}
    Right is at most 9.765625E-4
    F(4.8828125E-4)@11 = LineSearchPoint{point=PointSample{avg=13.55825399783879}, derivative=5680.504882383243}
    Right is at most 4.8828125E-4
    F(2.44140625E-4)@12 = LineSearchPoint{point=PointSample{avg=12.328223221397252}, derivative=24325.911
```
...[skipping 130909 bytes](etc/1.txt)...
```
    buffers unchanged since 0.0 => 0.3924290616764752
    F(0.0) = LineSearchPoint{point=PointSample{avg=0.3924290616764752}, derivative=-6093.5772849259265}
    New Minimum: 0.3924290616764752 > 0.3799796618303217
    F(1.1706346710892898E-5)@0 = LineSearchPoint{point=PointSample{avg=0.3799796618303217}, derivative=1727.8390587143645}
    F(5.853173355446449E-6) = LineSearchPoint{point=PointSample{avg=0.3805412358109025}, derivative=-2122.8341132117075}
    New Minimum: 0.3799796618303217 > 0.37886474662131286
    F(8.779760033169674E-6) = LineSearchPoint{point=PointSample{avg=0.37886474662131286}, derivative=-190.95242335816687}
    F(1.0243053372031285E-5) = LineSearchPoint{point=PointSample{avg=0.3790744510816736}, derivative=769.0403414987371}
    F(9.511406702600479E-6) = LineSearchPoint{point=PointSample{avg=0.37888257253410546}, derivative=289.3209711192243}
    End (narrow range) at 9.511406702600479E-6 to 8.779760033169674E-6
    Iteration 110 complete. Error: 0.37886474662131286 Total: 180095891276195.5300; Orientation: 0.0008; Line Search: 1.2138
    Returning cached value; 2 buffers unchanged since 0.0 => 0.40041137561549645
    F(0.0) = LineSearchPoint{point=PointSample{avg=0.40041137561549645}, derivative=-5000.367510042849}
    New Minimum: 0.40041137561549645 > 0.38627053499011776
    F(9.511406702600479E-6)@0 = LineSearchPoint{point=PointSample{avg=0.38627053499011776}, derivative=-1168.7307255482283}
    Right is at least 9.511406702600479E-6
    F(2.37785167565012E-5)@1 = LineSearchPoint{point=PointSample{avg=0.39866436278867323}, derivative=4563.802957256331}
    Right is at most 2.37785167565012E-5
    F(1.664496172955084E-5)@2 = LineSearchPoint{point=PointSample{avg=0.38740546979427287}, derivative=1663.8778971849122}
    Right is at most 1.664496172955084E-5
    New Minimum: 0.38627053499011776 > 0.3855853498165356
    F(1.307818421607566E-5)@3 = LineSearchPoint{point=PointSample{avg=0.3855853498165356}, derivative=244.81055652187433}
    Iteration 111 complete. Error: 0.3855853498165356 Total: 180097070759004.1000; Orientation: 0.0016; Line Search: 0.9505
    
```

Returns: 

```
    0.3855853498165356
```



Code from [MnistTestBase.java:131](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/mnist/MnistTestBase.java#L131) executed in 0.29 seconds: 
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
Code from [MnistTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/mnist/MnistTestBase.java#L144) executed in 0.49 seconds: 
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
      "SoftmaxActivationLayer/e1035fb9-1fe3-4846-a360-622900000004" : [ "java.util.HashMap", {
        "avgMsPerItem" : 0.00411587391424619,
        "medianMsPerItem" : "NaN",
        "avgMsPerItem_Backward" : 2.571864118948822E-6,
        "totalItems" : 723000,
        "backpropStatistics" : [ "java.util.HashMap", {
          "meanExponent" : -2.522604643878664,
          "tp50" : -0.0021860905600228062,
          "negative" : 500,
          "min" : -16.651869507246257,
          "max" : 0.0,
          "tp90" : -0.0020089318770016747,
          "mean" : -0.007205691139727786,
          "count" : 5000.0,
          "positive" : 0,
          "stdDev" : 0.2817698538715939,
          "tp75" : -0.002043912658376422,
          "zeros" : 4500
        } ],
        "totalBatches" : 1446,
        "class" : "com.simiacryptus.mindseye.layers.java.SoftmaxActivationLayer",
        "outputStatistics" : [ "java.util.HashMap", {
          "meanExponent" : -2.9000468422619035,
          "tp50" : 1.0100885743439188E-6,
          "negative" : 0,
          "min" : 1.7403612941300211E-9,
          "max" : 0.9913220404252625,
          "tp90" : 5.126011968231736E-6,
          "mean" : 0.1,
          "count" : 5000.0,
          "positive" : 5000,
          "stdDev" : 0.256410929654769,
          "tp75" : 3.267086862775743E-6,
          "zeros" : 0
        } ],
        "medianMsPerItem_Backward" : "NaN"
      } ],
      "BiasLayer/e1035fb9-1fe3-4846-a360-622900000002" : [ "java.util.HashMap", {
        "avgMsPerItem" : 0.022844520478561545,
        "medianMsPerItem" : "NaN",
        "avgMsPerItem_Backward" : 2.5853602987551835E-5,
        "totalItems" : 723000,
        "backpropStatistics" : [ "java.util.HashMap", {
          "meanExponent" : -7.478246313464634,
          "tp50" : -2.6330603325244865E-6,
          "negative" : 193687,
          "min" : -1.0404241095434342E-6,
          "max" : 1.0971098910685623E-6,
          "tp90" : -2.32336294805384E-6,
          "mean" : -7.392865923100477E-10,
          "count" : 392000.0,
          "positive" : 198313,
          "stdDev" : 3.4587926920444947E-7,
          "tp75" : -2.4237390254906025E-6,
          "zeros" : 0
        } ],
```
...[skipping 767 bytes](etc/2.txt)...
```
    
          "tp90" : -1.6381933206445066E-8,
          "mean" : 32.723533162978576,
          "count" : 392000.0,
          "positive" : 235603,
          "stdDev" : 77.92332480175311,
          "tp75" : -1.742470544472746E-8,
          "zeros" : 0
        } ],
        "medianMsPerItem_Backward" : "NaN"
      } ],
      "FullyConnectedLayer/e1035fb9-1fe3-4846-a360-622900000003" : [ "java.util.HashMap", {
        "avgMsPerItem" : 0.015233810735822936,
        "medianMsPerItem" : "NaN",
        "avgMsPerItem_Backward" : 7.818868527800832E-5,
        "totalItems" : 723000,
        "backpropStatistics" : [ "java.util.HashMap", {
          "meanExponent" : -5.69971205601174,
          "tp50" : -1.7024963505708066E-4,
          "negative" : 500,
          "min" : -0.001999759786731558,
          "max" : 0.0011807468441907356,
          "tp90" : -8.892165139024304E-6,
          "mean" : 8.719339291449861E-22,
          "count" : 5000.0,
          "positive" : 4500,
          "stdDev" : 2.5578283898040026E-4,
          "tp75" : -4.296921220822003E-5,
          "zeros" : 0
        } ],
        "totalBatches" : 1446,
        "weights" : [ "java.util.HashMap", {
          "tp50" : "NaN",
          "buffers" : 1,
          "max" : 0.002359276947594255,
          "tp90" : "NaN",
          "count" : 7840.0,
          "positive" : 4342,
          "tp75" : "NaN",
          "zeros" : 0,
          "meanExponent" : -3.6468795736571846,
          "negative" : 3498,
          "min" : -0.0020964245192659193,
          "mean" : 5.486546269691465E-5,
          "stdDev" : 4.30406010238426E-4
        } ],
        "class" : "com.simiacryptus.mindseye.layers.java.FullyConnectedLayer",
        "outputStatistics" : [ "java.util.HashMap", {
          "meanExponent" : 0.2553898587514713,
          "tp50" : -4.096168099170362,
          "negative" : 1872,
          "min" : -5.58876577055128,
          "max" : 11.563802831281444,
          "tp90" : -3.1789245283545853,
          "mean" : 1.2692908780025431,
          "count" : 5000.0,
          "positive" : 3128,
          "stdDev" : 3.5435776907454,
          "tp75" : -3.480634769220475,
          "zeros" : 0
        } ],
        "medianMsPerItem_Backward" : "NaN"
      } ]
    } ]
```



### Validation
If we run our model against the entire validation dataset, we get this accuracy:

Code from [MnistTestBase.java:201](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/mnist/MnistTestBase.java#L201) executed in 1.57 seconds: 
```java
    return MNIST.validationDataStream().mapToDouble(labeledObject ->
      predict(network, labeledObject)[0] == parse(labeledObject.label) ? 1 : 0)
      .average().getAsDouble() * 100;
```

Returns: 

```
    90.0990099009901
```



Let's examine some incorrectly predicted results in more detail:

Code from [MnistTestBase.java:208](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/mnist/MnistTestBase.java#L208) executed in 0.14 seconds: 
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
![[5]](etc/test.2.png)  | 6 (86.5%), 4 (5.4%), 2 (3.8%)  
![[4]](etc/test.3.png)  | 6 (42.5%), 0 (32.6%), 5 (12.8%)
![[6]](etc/test.4.png)  | 7 (42.0%), 6 (19.7%), 3 (12.4%)
![[9]](etc/test.5.png)  | 7 (55.8%), 9 (31.1%), 8 (9.1%) 
![[2]](etc/test.6.png)  | 7 (84.8%), 2 (8.5%), 9 (4.2%)  
![[7]](etc/test.7.png)  | 9 (51.6%), 7 (41.8%), 4 (3.7%) 
![[9]](etc/test.8.png)  | 4 (38.8%), 9 (29.1%), 8 (16.7%)
![[2]](etc/test.9.png)  | 8 (42.9%), 2 (25.0%), 7 (19.0%)
![[7]](etc/test.10.png) | 4 (65.5%), 9 (20.1%), 7 (10.7%)
![[2]](etc/test.11.png) | 9 (58.5%), 8 (13.4%), 4 (8.6%) 




