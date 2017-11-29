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
    PipelineNetwork/3cc8990a-29bd-4377-9ee9-863800000078
```



### Training
Training a model involves a few different components. First, our model is combined mapCoords a loss function. Then we take that model and combine it mapCoords our training data to define a trainable object. Finally, we use a simple iterative scheme to refine the weights of our model. The final output is the last output value of the loss function when evaluating the last batch.

Code from [L2NormalizationTest.java:47](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/opt/trainable/L2NormalizationTest.java#L47) executed in 180.08 seconds: 
```java
    SimpleLossNetwork supervisedNetwork = new SimpleLossNetwork(network, new EntropyLossLayer());
    Trainable trainable = new L12Normalizer(new SampledArrayTrainable(trainingData, supervisedNetwork, 1000)) {
      @Override
      protected double getL1(NNLayer layer) {
        return 0.0;
      }
      
      @Override
      protected double getL2(NNLayer layer) {
        return 1e4;
      }
    };
    return new IterativeTrainer(trainable)
      .setMonitor(monitor)
      .setTimeout(3, TimeUnit.MINUTES)
      .setMaxIterations(500)
      .run();
```
Logging: 
```
    LBFGS Accumulation History: 1 points
    Constructing line search parameters: GD
    Returning cached value; 2 buffers unchanged since 0.0 => 2.383014277029437
    th(0)=11.595327540167425;dx=-657417.4518396681
    Armijo: th(2.154434690031884)=3.051400525313487E10; dx=2.8326976229088383E10 delta=-3.051400524153954E10
    Armijo: th(1.077217345015942)=7.628348885980427E9; dx=1.4163346588385063E10 delta=-7.628348874385099E9
    Armijo: th(0.3590724483386473)=8.475265952927134E8; dx=4.720926827916184E9 delta=-8.475265836973859E8
    Armijo: th(0.08976811208466183)=5.295138930645068E7; dx=1.1800194188899064E9 delta=-5.295137771112314E7
    Armijo: th(0.017953622416932366)=2114024.6704118303; dx=2.3577745486081532E8 delta=-2114013.07508429
    Armijo: th(0.002992270402822061)=58051.08991474307; dx=3.906080375038567E7 delta=-58039.4945872029
    Armijo: th(4.2746720040315154E-4)=1108.3206544870814; dx=5353279.11723217 delta=-1096.725326946914
    Armijo: th(5.343340005039394E-5)=17.436798944940435; dx=526800.9336516462 delta=-5.84147140477301
    New Minimum: 11.595327540167425 > 8.901717274930398
    END: th(5.9370444500437714E-6)=8.901717274930398; dx=-295799.4535555938 delta=2.6936102652370266
    Iteration 1 complete. Error: 8.901717274930398 Total: 14988792343133.7300; Orientation: 0.0006; Line Search: 1.7672
    LBFGS Accumulation History: 1 points
    Returning cached value; 2 buffers unchanged since 0.0 => 1.778210958785291
    th(0)=8.936956477993085;dx=-539196.2700040567
    New Minimum: 8.936956477993085 > 8.671107402542622
    WOLF (strong): th(1.279097451943557E-5)=8.671107402542622; dx=324948.49418147246 delta=0.265849075450463
    New Minimum: 8.671107402542622 > 7.43713616935865
    WOLF (strong): th(6.395487259717785E-6)=7.43713616935865; dx=30653.555519882935 delta=1.499820308634435
    END: th(2.131829086572595E-6)=8.016679706547928; dx=-322232.15146836865 delta=0.9202767714451578
    Iteration 2 complete. Error: 7.43713616935865 Total: 14989578776825.2070; Orientation: 0.0008; Line Search: 0.5921
    LBFGS Accumulation History: 1 points
    Returning cached value; 2 buffe
```
...[skipping 178072 bytes](etc/1.txt)...
```
    6969579378176 delta=0.016743722112831705
    END: th(2.3279353767150057E-6)=2.4993030926217377; dx=-3786.5026026082846 delta=0.014559474152171514
    Iteration 322 complete. Error: 2.4971188446610775 Total: 15164522654335.0200; Orientation: 0.0006; Line Search: 0.3631
    LBFGS Accumulation History: 1 points
    Returning cached value; 2 buffers unchanged since 0.0 => 0.8977332089328903
    th(0)=2.519528970390258;dx=-16865.17417536146
    New Minimum: 2.519528970390258 > 2.502891257421332
    WOLF (strong): th(5.01538473174725E-6)=2.502891257421332; dx=8460.249996641896 delta=0.016637712968926355
    New Minimum: 2.502891257421332 > 2.495321944368285
    END: th(2.507692365873625E-6)=2.495321944368285; dx=-3455.796403308304 delta=0.02420702602197311
    Iteration 323 complete. Error: 2.495321944368285 Total: 15165074488572.4770; Orientation: 0.0006; Line Search: 0.3682
    LBFGS Accumulation History: 1 points
    Returning cached value; 2 buffers unchanged since 0.0 => 0.8566635818100745
    th(0)=2.4370056393662027;dx=-15904.364100903256
    New Minimum: 2.4370056393662027 > 2.4322033636009492
    WOLF (strong): th(5.402659424966265E-6)=2.4322033636009492; dx=15006.595627090797 delta=0.004802275765253494
    New Minimum: 2.4322033636009492 > 2.413678063237378
    END: th(2.7013297124831324E-6)=2.413678063237378; dx=-1154.8670024832873 delta=0.023327576128824745
    Iteration 324 complete. Error: 2.413678063237378 Total: 15165617864351.9260; Orientation: 0.0005; Line Search: 0.3654
    LBFGS Accumulation History: 1 points
    Returning cached value; 2 buffers unchanged since 0.0 => 0.8779582327946827
    th(0)=2.4802871570877096;dx=-12971.259463946768
    New Minimum: 2.4802871570877096 > 2.476840765309427
    WOLF (strong): th(5.819838441787515E-6)=2.476840765309427; dx=9401.0642561499 delta=0.0034463917782825426
    New Minimum: 2.476840765309427 > 2.46228305002175
    END: th(2.9099192208937575E-6)=2.46228305002175; dx=-1110.8126667145166 delta=0.018004107065959563
    Iteration 325 complete. Error: 2.46228305002175 Total: 15166155959758.3830; Orientation: 0.0007; Line Search: 0.3575
    
```

Returns: 

```
    2.483519520109833
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
Code from [MnistTestBase.java:154](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/mnist/MnistTestBase.java#L154) executed in 0.05 seconds: 
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
      "BiasLayer/3cc8990a-29bd-4377-9ee9-863800000079" : [ "java.util.HashMap", {
        "avgMsPerItem" : 0.014449256470164611,
        "medianMsPerItem" : "NaN",
        "avgMsPerItem_Backward" : 1.9442158555555544E-5,
        "totalItems" : 972000,
        "backpropStatistics" : [ "java.util.HashMap", {
          "meanExponent" : -8.625639879306878,
          "tp50" : -6.810265472839979E-7,
          "negative" : 148479,
          "min" : -4.821358167042221E-14,
          "max" : 4.908406577093637E-14,
          "tp90" : -6.260058664639151E-7,
          "mean" : 1.4597063593415273E-9,
          "count" : 392000.0,
          "positive" : 243521,
          "stdDev" : 1.1821760627068128E-7,
          "tp75" : -6.415644751327735E-7,
          "zeros" : 0
        } ],
        "totalBatches" : 1944,
        "weights" : [ "java.util.HashMap", {
          "tp50" : "NaN",
          "buffers" : 1,
          "max" : 1.1705770523264429E-8,
          "tp90" : "NaN",
          "count" : 784.0,
          "positive" : 329,
          "tp75" : "NaN",
          "zeros" : 0,
          "meanExponent" : -9.073965580559234,
          "negative" : 455,
          "min" : -9.116108120208312E-9,
          "mean" : -5.354774227886841E-10,
          "stdDev" : 2.608654116850969E-9
        } ],
        "class" : "com.simiacryptus.mindseye.layers.java.BiasLayer",
        "outputStatistics" : [ "java.util.HashMap", {
          "meanExponent" : -7.0590947230243035,
          "tp50" : -8.683581242629615E-9,
          "negative" : 179181,
          "min" : 7.545091542217911E-10,
          "max" : 7.545091542217911E-10,
          "tp90" : -8.683581242629615E-9,
          "mean" : 32.64019387701577,
          "count" : 392000.0,
          "positive" : 212819,
          "stdDev" : 77.98068290848863,
          "tp75" : -8.683581242629615E-9,
          "zeros" : 0
        } ],
        "medianMsPerItem_Backward" : "NaN"
      } ],
      "FullyConnectedLayer/3cc8990a-29bd-4377-9ee9-86380000007a" : [ "java.util.HashMap", {
        "avgMsPerItem" : 0.008918245718106996,
        "medianMsPerItem" : "NaN",
        "avgMsPerItem_Backward" : 5.063539760905353E-5,
        "totalItems" : 972000,
        "
```
...[skipping 781 bytes](etc/2.txt)...
```
    95,
          "negative" : 5060,
          "min" : -4.226750074424728E-4,
          "mean" : 2.5808131722319528E-12,
          "stdDev" : 9.610578466013971E-5
        } ],
        "class" : "com.simiacryptus.mindseye.layers.java.FullyConnectedLayer",
        "outputStatistics" : [ "java.util.HashMap", {
          "meanExponent" : -0.27284502387573223,
          "tp50" : -1.5905234385243476,
          "negative" : 2736,
          "min" : -2.730919552747875,
          "max" : 2.830697255065983,
          "tp90" : -1.3302091747853861,
          "mean" : 6.073229208924461E-8,
          "count" : 5000.0,
          "positive" : 2264,
          "stdDev" : 1.1093397028109473,
          "tp75" : -1.4065330115946915,
          "zeros" : 0
        } ],
        "medianMsPerItem_Backward" : "NaN"
      } ],
      "SoftmaxActivationLayer/3cc8990a-29bd-4377-9ee9-86380000007b" : [ "java.util.HashMap", {
        "avgMsPerItem" : 0.002155962548353901,
        "medianMsPerItem" : "NaN",
        "avgMsPerItem_Backward" : 1.2325429444444462E-6,
        "totalItems" : 972000,
        "backpropStatistics" : [ "java.util.HashMap", {
          "meanExponent" : -2.308452897321443,
          "tp50" : -0.004241682280613799,
          "negative" : 500,
          "min" : -0.05384834918398296,
          "max" : 0.0,
          "tp90" : -0.002745703064715424,
          "mean" : -6.217558092386427E-4,
          "count" : 5000.0,
          "positive" : 0,
          "stdDev" : 0.00285869990323643,
          "tp75" : -0.0032137175322187487,
          "zeros" : 4500
        } ],
        "totalBatches" : 1944,
        "class" : "com.simiacryptus.mindseye.layers.java.SoftmaxActivationLayer",
        "outputStatistics" : [ "java.util.HashMap", {
          "meanExponent" : -1.3087017653859707,
          "tp50" : 0.007581909740478742,
          "negative" : 0,
          "min" : 8.608309424019718E-4,
          "max" : 0.5555024381037329,
          "tp90" : 0.010688198096321886,
          "mean" : 0.1,
          "count" : 5000.0,
          "positive" : 5000,
          "stdDev" : 0.14839734443481534,
          "tp75" : 0.009386988337774526,
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
    85.78
```



Let's examine some incorrectly predicted results in more detail:

Code from [MnistTestBase.java:218](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/mnist/MnistTestBase.java#L218) executed in 0.05 seconds: 
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
![[5]](etc/test.2.png)  | 6 (21.0%), 4 (18.2%), 2 (16.2%)
![[9]](etc/test.3.png)  | 7 (30.5%), 9 (26.8%), 4 (12.6%)
![[4]](etc/test.4.png)  | 0 (28.9%), 6 (27.6%), 4 (14.2%)
![[2]](etc/test.5.png)  | 3 (25.0%), 2 (16.8%), 8 (12.8%)
![[5]](etc/test.6.png)  | 3 (30.8%), 5 (29.4%), 8 (15.4%)
![[1]](etc/test.7.png)  | 3 (21.0%), 5 (13.5%), 1 (13.1%)
![[5]](etc/test.8.png)  | 3 (21.7%), 5 (21.1%), 8 (13.4%)
![[6]](etc/test.9.png)  | 2 (25.2%), 6 (22.5%), 4 (14.5%)
![[3]](etc/test.10.png) | 2 (27.6%), 3 (18.7%), 8 (11.4%)
![[4]](etc/test.11.png) | 9 (17.3%), 4 (16.6%), 5 (14.5%)




