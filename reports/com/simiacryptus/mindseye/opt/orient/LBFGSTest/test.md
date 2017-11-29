### Model
This is a very simple model that performs basic logistic regression. It is expected to be trainable to about 91% accuracy on MNIST.

Code from [MnistTestBase.java:295](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/mnist/MnistTestBase.java#L295) executed in 0.00 seconds: 
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
    PipelineNetwork/3cc8990a-29bd-4377-9ee9-863800000025
```



### Training
Code from [LBFGSTest.java:45](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/opt/orient/LBFGSTest.java#L45) executed in 308.51 seconds: 
```java
    SimpleLossNetwork supervisedNetwork = new SimpleLossNetwork(network, new EntropyLossLayer());
    return new ValidatingTrainer(
      new SampledArrayTrainable(trainingData, supervisedNetwork, 1000, 10000),
      new ArrayTrainable(trainingData, supervisedNetwork).cached()
    )
      .setMonitor(monitor)
      //.setOrientation(new ValidatingOrientationWrapper(new LBFGS()))
      .setOrientation(new LBFGS())
      .setLineSearchFactory(name -> name.contains("LBFGS") ? new QuadraticSearch().setCurrentRate(1.0) : new QuadraticSearch())
      .setTimeout(5, TimeUnit.MINUTES)
      .setMaxIterations(500)
      .run();
```
Logging: 
```
    Epoch parameters: 1000, 1
    Phase 0: TrainingPhase{trainingSubject=SampledCachedTrainable{inner=PerformanceWrapper{inner=SampledArrayTrainable{inner=ArrayTrainable{inner=com.simiacryptus.mindseye.eval.GpuTrainable@79fcd194}}}}, orientation=com.simiacryptus.mindseye.opt.orient.LBFGS@73d8c8d7}
    resetAndMeasure; trainingSize=1000
    LBFGS Accumulation History: 1 points
    Constructing line search parameters: GD
    Returning cached value; 2 buffers unchanged since 0.0 => 2.5353334640831187
    F(0.0) = LineSearchPoint{point=PointSample{avg=2.5353334640831187}, derivative=-401818.244097504}
    New Minimum: 2.5353334640831187 > 2.535313298815601
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=2.535313298815601}, derivative=-401811.81391421385}, delta = -2.0165267517580077E-5
    New Minimum: 2.535313298815601 > 2.5351923139871717
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=2.5351923139871717}, derivative=-401773.232662213}, delta = -1.4115009594695138E-4
    New Minimum: 2.5351923139871717 > 2.534345745474667
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=2.534345745474667}, derivative=-401503.1565743357}, delta = -9.877186084517042E-4
    New Minimum: 2.534345745474667 > 2.5284357069524326
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=2.5284357069524326}, derivative=-399612.27365445637}, delta = -0.006897757130686077
    New Minimum: 2.5284357069524326 > 2.4878471569884946
    F(2.4010000000000004E-7) = LineSearchPoint{point=PointSample{avg=2.4878471569884946}, derivative=-386361.8304405435}, delta = -0.04748630709462409
    New Minimum: 2.4878471569884946 > 2.242050422089372
    F(1.6807000000000003E-6) = LineSearchPoint{point=PointSample{avg=2.242050422089372}, derivative=-293778.54507738026}, delta = -0.29328304199374644
    New Minimum: 2.242050422089372 > 2.122690267578487
    F(1.1764900000000001E-5) = LineSearchPoint{point=PointSample{avg=2.122690267578487}, derivative=171799.74622042512}, delta = -0.41264319650463177
    2.122690267578487 <= 2.5353334640831187
    New Minimum: 2.1226902675784
```
...[skipping 303166 bytes](etc/1.txt)...
```
    }, derivative=-718.3793426977035}
    New Minimum: 0.35814812087557046 > 0.35615682825626793
    F(7.523708843400783E-6) = LineSearchPoint{point=PointSample{avg=0.35615682825626793}, derivative=-308.3889410789129}, delta = -0.0019912926193025315
    F(5.266596190380548E-5) = LineSearchPoint{point=PointSample{avg=0.3762273467841313}, derivative=2106.3397405258597}, delta = 0.018079225908560814
    F(5.266596190380548E-7) = LineSearchPoint{point=PointSample{avg=0.35795829629224124}, derivative=-689.4941602291179}, delta = -1.8982458332922336E-4
    F(3.6866173332663835E-6) = LineSearchPoint{point=PointSample{avg=0.35697914577012735}, derivative=-516.8001802272545}, delta = -0.0011689751054431152
    F(2.5806321332864685E-5) = LineSearchPoint{point=PointSample{avg=0.35770123924174124}, derivative=671.9961576847161}, delta = -4.468816338292214E-4
    0.35770123924174124 <= 0.35814812087557046
    New Minimum: 0.35615682825626793 > 0.3556721610146599
    Returning cached value; 2 buffers unchanged since 0.0 => 0.3556721610146599
    isLeft=false; isBracketed=true; leftPoint=LineSearchPoint{point=PointSample{avg=0.35814812087557046}, derivative=-718.3793426977035}; rightPoint=LineSearchPoint{point=PointSample{avg=0.35770123924174124}, derivative=671.9961576847161}
    F(1.3333612503564699E-5) = LineSearchPoint{point=PointSample{avg=0.3556721610146599}, derivative=4.9673179521066375}, delta = -0.002475959860910537
    Right bracket at 1.3333612503564699E-5
    Converged to right
    Returning cached value; 2 buffers unchanged since 1.3333612503564699E-5 => 0.3556721610146599
    Overall network state change: {FullyConnectedLayer=0.9982048146352306, BiasLayer=0.9905946288234951}
    Iteration 127 complete. Error: 0.3556721610146599 (3853 in 2.724 seconds; 0.001 in orientation, 0.421 in gc, 2.723 in line search; 2.716 eval time)
    Training timeout
    Epoch 19 result with 9 iterations, 3853/1000000 samples: {validation *= 2^-0.04136; training *= 2^-0.108; Overtraining = 2.62}, {itr*=1.92, len*=1.15} 0 since improvement; 5.9450 validation time
    Training 19 runPhase halted
    
```

Returns: 

```
    0.3623909480464067
```



Code from [MnistTestBase.java:141](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/mnist/MnistTestBase.java#L141) executed in 0.01 seconds: 
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
Code from [MnistTestBase.java:154](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/mnist/MnistTestBase.java#L154) executed in 4.75 seconds: 
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
      "BiasLayer/3cc8990a-29bd-4377-9ee9-863800000026" : [ "java.util.HashMap", {
        "avgMsPerItem" : 0.014932056310908104,
        "medianMsPerItem" : "NaN",
        "avgMsPerItem_Backward" : 7.2001112603521816E-6,
        "totalItems" : 2430506,
        "backpropStatistics" : [ "java.util.HashMap", {
          "meanExponent" : -9.330101534725822,
          "tp50" : -4.452269357967089E-8,
          "negative" : 11499986,
          "min" : -2.8697819436418478E-8,
          "max" : 2.7793877013264805E-8,
          "tp90" : -3.894308322675058E-8,
          "mean" : -7.773078426753153E-13,
          "count" : 2.352E7,
          "positive" : 12020014,
          "stdDev" : 5.673619212921973E-9,
          "tp75" : -4.069195796788283E-8,
          "zeros" : 0
        } ],
        "totalBatches" : 1530,
        "weights" : [ "java.util.HashMap", {
          "tp50" : "NaN",
          "buffers" : 1,
          "max" : 1.8680267211747442E-8,
          "tp90" : "NaN",
          "count" : 784.0,
          "positive" : 363,
          "tp75" : "NaN",
          "zeros" : 0,
          "meanExponent" : -8.565868204737942,
          "negative" : 421,
          "min" : -1.7107356734005886E-8,
          "mean" : -4.09212843346153E-10,
          "stdDev" : 5.3121850968680586E-9
        } ],
        "class" : "com.simiacryptus.mindseye.layers.java.BiasLayer",
        "outputStatistics" : [ "java.util.HashMap", {
          "meanExponent" : -6.54491991950254,
          "tp50" : -1.7107356734005886E-8,
          "negative" : 10078008,
          "min" : -1.9336191699787593E-9,
          "max" : -1.9336191699787593E-9,
          "tp90" : -1.6578206336798823E-8,
          "mean" : 33.05147801829151,
          "count" : 2.352E7,
          "positive" : 13441992,
          "stdDev" : 78.21283592102374,
          "tp75" : -1.6578206336798823E-8,
          "zeros" : 0
        } ],
        "medianMsPerItem_Backward" : "NaN"
      } ],
      "SoftmaxActivationLayer/3cc8990a-29bd-4377-9ee9-863800000028" : [ "java.util.HashMap", {
        "avgMsPerItem" : 0.0025693321403033004,
        "medianMsPerItem" : "NaN",
        "avgMsPerItem_Backward" : 3.8059509499077073E-7,
        "totalItems" 
```
...[skipping 806 bytes](etc/2.txt)...
```
    8,
          "tp90" : 3.6179199339978386E-6,
          "mean" : 0.1,
          "count" : 300000.0,
          "positive" : 300000,
          "stdDev" : 0.26116597745366665,
          "tp75" : 2.1843528047242247E-6,
          "zeros" : 0
        } ],
        "medianMsPerItem_Backward" : "NaN"
      } ],
      "FullyConnectedLayer/3cc8990a-29bd-4377-9ee9-863800000027" : [ "java.util.HashMap", {
        "avgMsPerItem" : 0.008876063512289204,
        "medianMsPerItem" : "NaN",
        "avgMsPerItem_Backward" : 1.6593911239047965E-5,
        "totalItems" : 2430506,
        "backpropStatistics" : [ "java.util.HashMap", {
          "meanExponent" : -7.569059619752532,
          "tp50" : -2.19616854684518E-6,
          "negative" : 30000,
          "min" : -3.33296381803171E-5,
          "max" : 3.2545090733016824E-5,
          "tp90" : -1.268151438452899E-7,
          "mean" : 3.086732108213691E-24,
          "count" : 300000.0,
          "positive" : 270000,
          "stdDev" : 4.14340519649458E-6,
          "tp75" : -5.674817016893995E-7,
          "zeros" : 0
        } ],
        "totalBatches" : 1530,
        "weights" : [ "java.util.HashMap", {
          "tp50" : "NaN",
          "buffers" : 1,
          "max" : 0.00213760398860907,
          "tp90" : "NaN",
          "count" : 7840.0,
          "positive" : 4216,
          "tp75" : "NaN",
          "zeros" : 0,
          "meanExponent" : -3.635498283821594,
          "negative" : 3624,
          "min" : -0.0019396087292744952,
          "mean" : 4.596400686998443E-5,
          "stdDev" : 4.3941706790268147E-4
        } ],
        "class" : "com.simiacryptus.mindseye.layers.java.FullyConnectedLayer",
        "outputStatistics" : [ "java.util.HashMap", {
          "meanExponent" : 0.25544898434834556,
          "tp50" : -4.423806457379767,
          "negative" : 116625,
          "min" : -9.343026813487734,
          "max" : 13.14270425875519,
          "tp90" : -3.4450287984503203,
          "mean" : 1.10126944590862,
          "count" : 300000.0,
          "positive" : 183375,
          "stdDev" : 3.599329685784909,
          "tp75" : -3.7689287360193267,
          "zeros" : 0
        } ],
        "medianMsPerItem_Backward" : "NaN"
      } ]
    } ]
```



### Validation
If we run our model against the entire validation dataset, we get this accuracy:

Code from [MnistTestBase.java:211](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/mnist/MnistTestBase.java#L211) executed in 0.60 seconds: 
```java
    return MNIST.validationDataStream().mapToDouble(labeledObject ->
      predict(network, labeledObject)[0] == parse(labeledObject.label) ? 1 : 0)
      .average().getAsDouble() * 100;
```

Returns: 

```
    90.48
```



Let's examine some incorrectly predicted results in more detail:

Code from [MnistTestBase.java:218](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/mnist/MnistTestBase.java#L218) executed in 0.23 seconds: 
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
![[5]](etc/test.2.png)  | 6 (87.4%), 2 (8.8%), 4 (2.0%)  
![[4]](etc/test.3.png)  | 0 (46.8%), 6 (29.9%), 4 (8.0%) 
![[3]](etc/test.4.png)  | 2 (52.7%), 3 (40.5%), 8 (4.3%) 
![[2]](etc/test.5.png)  | 7 (41.1%), 2 (41.1%), 9 (14.7%)
![[9]](etc/test.6.png)  | 4 (35.1%), 9 (20.2%), 8 (19.8%)
![[7]](etc/test.7.png)  | 4 (62.0%), 9 (29.0%), 7 (5.2%) 
![[2]](etc/test.8.png)  | 9 (53.6%), 8 (13.4%), 4 (11.5%)
![[9]](etc/test.9.png)  | 4 (36.2%), 9 (32.4%), 3 (21.9%)
![[3]](etc/test.10.png) | 8 (30.6%), 3 (19.5%), 5 (15.4%)
![[6]](etc/test.11.png) | 5 (75.8%), 6 (11.5%), 8 (7.2%) 




