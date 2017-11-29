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
    PipelineNetwork/3cc8990a-29bd-4377-9ee9-863800000049
```



### Training
Code from [LinearSumConstraintTest.java:43](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/opt/region/LinearSumConstraintTest.java#L43) executed in 182.23 seconds: 
```java
    SimpleLossNetwork supervisedNetwork = new SimpleLossNetwork(network, new EntropyLossLayer());
    Trainable trainable = new SampledArrayTrainable(trainingData, supervisedNetwork, 10000);
    TrustRegionStrategy trustRegionStrategy = new TrustRegionStrategy() {
      @Override
      public TrustRegion getRegionPolicy(NNLayer layer) {
        return new LinearSumConstraint();
      }
    };
    return new IterativeTrainer(trainable)
      .setIterationsPerSample(100)
      .setMonitor(monitor)
      //.setOrientation(new ValidatingOrientationWrapper(trustRegionStrategy))
      .setOrientation(trustRegionStrategy)
      .setTimeout(3, TimeUnit.MINUTES)
      .setMaxIterations(500)
      .run();
```
Logging: 
```
    LBFGS Accumulation History: 1 points
    Constructing line search parameters: GD+Trust
    Returning cached value; 2 buffers unchanged since 0.0 => 2.433649456323333
    th(0)=2.433649456323333;dx=-415005.08950926317
    Armijo: th(2.154434690031884)=19.645656013425196; dx=-2.556137595321686E-17 delta=-17.212006557101862
    Armijo: th(1.077217345015942)=19.19250726712397; dx=2.5663169286857956E-18 delta=-16.758857810800635
    Armijo: th(0.3590724483386473)=19.374202351836992; dx=0.012311230599118844 delta=-16.940552895513658
    Armijo: th(0.08976811208466183)=19.602089961795315; dx=0.2779053745397098 delta=-17.16844050547198
    Armijo: th(0.017953622416932366)=19.587523419515904; dx=5.678392138766885 delta=-17.15387396319257
    Armijo: th(0.002992270402822061)=19.42517231495026; dx=182.76966271419465 delta=-16.991522858626926
    Armijo: th(4.2746720040315154E-4)=16.20132177661998; dx=32106.944684350707 delta=-13.767672320296647
    Armijo: th(5.343340005039394E-5)=2.46931002224201; dx=63878.4349582583 delta=-0.03566056591867728
    New Minimum: 2.433649456323333 > 2.2464475881108794
    END: th(5.9370444500437714E-6)=2.2464475881108794; dx=-55897.591070996285 delta=0.18720186821245344
    Iteration 1 complete. Error: 2.2464475881108794 Total: 13963486722845.1110; Orientation: 0.0034; Line Search: 8.2756
    LBFGS Accumulation History: 2 points
    Returning cached value; 2 buffers unchanged since 0.0 => 2.2464475881108794
    th(0)=2.2464475881108794;dx=-312342.4119629793
    New Minimum: 2.2464475881108794 > 1.8142361764058053
    WOLF (strong): th(1.279097451943557E-5)=1.8142361764058053; dx=119548.20066142976 delta=0.43221141170507416
    New Minimum: 1.8142361764058053 > 1.6719242904506664
    END: th(6.395487259717785E-6)=1.6719242904506664; dx=-52503.82974295158 delta=0.574523297660213
    Iteration 2 complete. Error: 1.6719242904506664 Total: 13965487937414.5140; Orientation: 0.0006; Line Search: 2.0005
    LBFGS Accumulation History: 3 points
    Returning cached value; 2 buffers unchanged since 0.0 => 1.6719242904506664
    th(0)=1.6719242904506664;dx=-357970.7469841741
```
...[skipping 59478 bytes](etc/1.txt)...
```
    837165E-4
    Iteration 91 complete. Error: 0.3328550775135105 Total: 14130213473806.7620; Orientation: 0.0009; Line Search: 0.9736
    Orientation vanished. Popping history element from 0.33586187724848693, 0.33438682179598817, 0.3335096896226942, 0.3328550775135105
    LBFGS Accumulation History: 3 points
    Returning cached value; 2 buffers unchanged since 0.0 => 0.3328550775135105
    th(0)=0.3328550775135105;dx=-189.34852494332358
    New Minimum: 0.3328550775135105 > 0.331500078278377
    END: th(1.964100856443076E-5)=0.331500078278377; dx=-137.02424910968156 delta=0.0013549992351334916
    Iteration 92 complete. Error: 0.331500078278377 Total: 14131197553460.7870; Orientation: 0.0014; Line Search: 0.9825
    Orientation vanished. Popping history element from 0.33438682179598817, 0.3335096896226942, 0.3328550775135105, 0.331500078278377
    LBFGS Accumulation History: 3 points
    Returning cached value; 2 buffers unchanged since 0.0 => 0.331500078278377
    th(0)=0.331500078278377;dx=-184.88332451881863
    New Minimum: 0.331500078278377 > 0.3287377678683504
    END: th(4.231527019842296E-5)=0.3287377678683504; dx=-126.38079484470623 delta=0.002762310410026614
    Iteration 93 complete. Error: 0.3287377678683504 Total: 14132165402142.8030; Orientation: 0.0011; Line Search: 0.9666
    Orientation vanished. Popping history element from 0.3335096896226942, 0.3328550775135105, 0.331500078278377, 0.3287377678683504
    LBFGS Accumulation History: 3 points
    Returning cached value; 2 buffers unchanged since 0.0 => 0.3287377678683504
    th(0)=0.3287377678683504;dx=-198.72525577304202
    Armijo: th(9.116548603355478E-5)=0.33077194221678063; dx=232.85097018500517 delta=-0.002034174348430229
    New Minimum: 0.3287377678683504 > 0.3278568419760609
    WOLF (strong): th(4.558274301677739E-5)=0.3278568419760609; dx=56.758361758352464 delta=8.809258922894969E-4
    END: th(1.519424767225913E-5)=0.32795131861855875; dx=-71.97430093019959 delta=7.864492497916542E-4
    Iteration 94 complete. Error: 0.3278568419760609 Total: 14135108525772.8340; Orientation: 0.0015; Line Search: 2.9415
    
```

Returns: 

```
    0.3278568419760609
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
Code from [MnistTestBase.java:154](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/mnist/MnistTestBase.java#L154) executed in 0.55 seconds: 
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
      "BiasLayer/3cc8990a-29bd-4377-9ee9-86380000004a" : [ "java.util.HashMap", {
        "avgMsPerItem" : 0.012433837360869565,
        "medianMsPerItem" : "NaN",
        "avgMsPerItem_Backward" : 2.9832004897826106E-6,
        "totalItems" : 1840000,
        "backpropStatistics" : [ "java.util.HashMap", {
          "meanExponent" : -8.67914249886011,
          "tp50" : -2.349909754557481E-7,
          "negative" : 1876192,
          "min" : -8.812828955119639E-8,
          "max" : 7.4962830250826E-8,
          "tp90" : -2.037024543097599E-7,
          "mean" : -2.7028671127914617E-11,
          "count" : 3920000.0,
          "positive" : 2043808,
          "stdDev" : 2.7495036671763147E-8,
          "tp75" : -2.1333318079161294E-7,
          "zeros" : 0
        } ],
        "totalBatches" : 368,
        "weights" : [ "java.util.HashMap", {
          "tp50" : "NaN",
          "buffers" : 1,
          "max" : 9.002328270856454E-9,
          "tp90" : "NaN",
          "count" : 784.0,
          "positive" : 391,
          "tp75" : "NaN",
          "zeros" : 0,
          "meanExponent" : -9.832878770859363,
          "negative" : 393,
          "min" : -1.3440107133178546E-8,
          "mean" : -1.4088222191961346E-10,
          "stdDev" : 2.031898876575874E-9
        } ],
        "class" : "com.simiacryptus.mindseye.layers.java.BiasLayer",
        "outputStatistics" : [ "java.util.HashMap", {
          "meanExponent" : -7.602958311948974,
          "tp50" : -1.3440107133178546E-8,
          "negative" : 1574693,
          "min" : -1.2207470406512945E-11,
          "max" : -1.2207470406512945E-11,
          "tp90" : -1.2545520306356636E-8,
          "mean" : 33.1147438774133,
          "count" : 3920000.0,
          "positive" : 2345307,
          "stdDev" : 78.3174209296068,
          "tp75" : -1.2545520306356636E-8,
          "zeros" : 0
        } ],
        "medianMsPerItem_Backward" : "NaN"
      } ],
      "FullyConnectedLayer/3cc8990a-29bd-4377-9ee9-86380000004b" : [ "java.util.HashMap", {
        "avgMsPerItem" : 0.007937926790217395,
        "medianMsPerItem" : "NaN",
        "avgMsPerItem_Backward" : 6.005000778695653E-6,
        "totalItems" : 18400
```
...[skipping 792 bytes](etc/2.txt)...
```
    912743,
          "negative" : 3546,
          "min" : -0.001884551976097795,
          "mean" : 4.095355304791772E-5,
          "stdDev" : 3.6994173117534E-4
        } ],
        "class" : "com.simiacryptus.mindseye.layers.java.FullyConnectedLayer",
        "outputStatistics" : [ "java.util.HashMap", {
          "meanExponent" : 0.24221994164593924,
          "tp50" : -3.975924361354311,
          "negative" : 18715,
          "min" : -6.154015631138495,
          "max" : 10.781463900427909,
          "tp90" : -3.1142375747875186,
          "mean" : 1.176302937213581,
          "count" : 50000.0,
          "positive" : 31285,
          "stdDev" : 3.395986027684285,
          "tp75" : -3.400982541880216,
          "zeros" : 0
        } ],
        "medianMsPerItem_Backward" : "NaN"
      } ],
      "SoftmaxActivationLayer/3cc8990a-29bd-4377-9ee9-86380000004c" : [ "java.util.HashMap", {
        "avgMsPerItem" : 0.00198747243152174,
        "medianMsPerItem" : "NaN",
        "avgMsPerItem_Backward" : 9.183029076086958E-8,
        "totalItems" : 1840000,
        "backpropStatistics" : [ "java.util.HashMap", {
          "meanExponent" : -3.556823678972538,
          "tp50" : -2.1539435179272984E-4,
          "negative" : 5000,
          "min" : -2.6660987798104134,
          "max" : 0.0,
          "tp90" : -2.0109966503554448E-4,
          "mean" : -0.0010619575034677658,
          "count" : 50000.0,
          "positive" : 0,
          "stdDev" : 0.1890447251060161,
          "tp75" : -2.045388725829398E-4,
          "zeros" : 45000
        } ],
        "totalBatches" : 368,
        "class" : "com.simiacryptus.mindseye.layers.java.SoftmaxActivationLayer",
        "outputStatistics" : [ "java.util.HashMap", {
          "meanExponent" : -2.838493281171822,
          "tp50" : 1.7547088347427903E-6,
          "negative" : 0,
          "min" : 3.0796538319849784E-9,
          "max" : 0.9944600074068468,
          "tp90" : 7.302588133059375E-6,
          "mean" : 0.1,
          "count" : 50000.0,
          "positive" : 50000,
          "stdDev" : 0.2593090207945561,
          "tp75" : 4.665286082050566E-6,
          "zeros" : 0
        } ],
        "medianMsPerItem_Backward" : "NaN"
      } ]
    } ]
```



### Validation
If we run our model against the entire validation dataset, we get this accuracy:

Code from [MnistTestBase.java:211](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/mnist/MnistTestBase.java#L211) executed in 0.67 seconds: 
```java
    return MNIST.validationDataStream().mapToDouble(labeledObject ->
      predict(network, labeledObject)[0] == parse(labeledObject.label) ? 1 : 0)
      .average().getAsDouble() * 100;
```

Returns: 

```
    90.53
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
![[5]](etc/test.2.png)  | 6 (73.0%), 2 (14.9%), 4 (4.9%) 
![[4]](etc/test.3.png)  | 6 (50.4%), 0 (32.9%), 4 (5.7%) 
![[3]](etc/test.4.png)  | 2 (52.7%), 3 (36.9%), 8 (5.9%) 
![[6]](etc/test.5.png)  | 2 (28.4%), 7 (22.9%), 6 (17.7%)
![[2]](etc/test.6.png)  | 7 (85.0%), 2 (8.3%), 9 (4.7%)  
![[9]](etc/test.7.png)  | 4 (35.6%), 9 (29.4%), 8 (19.8%)
![[7]](etc/test.8.png)  | 1 (55.1%), 7 (24.6%), 9 (7.0%) 
![[7]](etc/test.9.png)  | 4 (69.3%), 9 (22.6%), 7 (5.2%) 
![[2]](etc/test.10.png) | 9 (56.6%), 8 (11.6%), 4 (9.1%) 
![[9]](etc/test.11.png) | 3 (44.6%), 4 (25.9%), 9 (20.8%)




