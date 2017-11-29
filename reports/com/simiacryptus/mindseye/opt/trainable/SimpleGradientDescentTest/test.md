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
    PipelineNetwork/3cc8990a-29bd-4377-9ee9-863800000081
```



### Training
Training a model involves a few different components. First, our model is combined mapCoords a loss function. Then we take that model and combine it mapCoords our training data to define a trainable object. Finally, we use a simple iterative scheme to refine the weights of our model. The final output is the last output value of the loss function when evaluating the last batch.

Code from [SimpleGradientDescentTest.java:50](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/opt/trainable/SimpleGradientDescentTest.java#L50) executed in 181.29 seconds: 
```java
    SimpleLossNetwork supervisedNetwork = new SimpleLossNetwork(network, new EntropyLossLayer());
    ArrayList<Tensor[]> trainingList = new ArrayList<>(Arrays.stream(trainingData).collect(Collectors.toList()));
    Collections.shuffle(trainingList);
    Tensor[][] randomSelection = trainingList.subList(0, 10000).toArray(new Tensor[][]{});
    Trainable trainable = new ArrayTrainable(randomSelection, supervisedNetwork);
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
    th(0)=2.4504492653349077;dx=-327533.21988441446
    Armijo: th(2.154434690031884)=11.020623457678447; dx=0.0012666324849328382 delta=-8.57017419234354
    Armijo: th(1.077217345015942)=11.01993886803358; dx=0.0012750824325310712 delta=-8.569489602698674
    Armijo: th(0.3590724483386473)=11.015714041332947; dx=0.05774979003895251 delta=-8.565264775998038
    Armijo: th(0.08976811208466183)=11.000502550879533; dx=0.5033372562539681 delta=-8.550053285544625
    Armijo: th(0.017953622416932366)=10.938889345677978; dx=9.358589053290565 delta=-8.488440080343072
    Armijo: th(0.002992270402822061)=10.432242538809971; dx=465.8518659802253 delta=-7.981793273475064
    Armijo: th(4.2746720040315154E-4)=7.070033302144018; dx=15855.129251168328 delta=-4.61958403680911
    New Minimum: 2.4504492653349077 > 1.520936649574149
    WOLF (strong): th(5.343340005039394E-5)=1.520936649574149; dx=36413.98310694243 delta=0.9295126157607587
    END: th(5.9370444500437714E-6)=1.7513879964899703; dx=-164977.02481979417 delta=0.6990612688449374
    Iteration 1 complete. Error: 1.520936649574149 Total: 15178780480706.5430; Orientation: 0.0011; Line Search: 9.1118
    LBFGS Accumulation History: 1 points
    th(0)=1.7513879964899703;dx=-218195.9956320631
    New Minimum: 1.7513879964899703 > 1.6865173421271418
    WOLF (strong): th(1.279097451943557E-5)=1.6865173421271418; dx=158613.5268464082 delta=0.06487065436282857
    New Minimum: 1.6865173421271418 > 1.4039519241520857
    WOLF (strong): th(6.395487259717785E-6)=1.4039519241520857; dx=1868.6196150157928 delta=0.34743607233788465
    END: th(2.131829086572595E-6)=1.5559060438042338; dx=-146823.0930486161 delta=0.19548195268573654
    Iteration 2 complete. Error: 1.4039519241520857 Total: 15183720269788.2190; Orientation: 0.0007; Line Search: 3.9447
    LBFGS Accumulation History: 1 points
    th(0)=1.5559060438042338;dx=-131425.25845112203
    New Minimum: 1.5559060438042338 > 1.300289247745035
    END: th(4.592886537330983E-6)=1.300289247745035; dx=-90570.53965587995 delta
```
...[skipping 18089 bytes](etc/1.txt)...
```
    135371924 Total: 15332236091949.6680; Orientation: 0.0007; Line Search: 1.9185
    LBFGS Accumulation History: 1 points
    th(0)=0.39091051135371924;dx=-3208.922558008966
    Armijo: th(6.455668110908157E-5)=1.0846282126192897; dx=44358.7268609072 delta=-0.6937177012655704
    Armijo: th(3.227834055454078E-5)=0.5350972233483442; dx=22094.04396482954 delta=-0.14418671199462496
    Armijo: th(1.0759446851513594E-5)=0.3938775154432871; dx=4434.792529304121 delta=-0.0029670040895678484
    New Minimum: 0.39091051135371924 > 0.3878335255338854
    END: th(2.6898617128783985E-6)=0.3878335255338854; dx=-1362.8223484276152 delta=0.003076985819833844
    Iteration 41 complete. Error: 0.3878335255338854 Total: 15338051866781.7640; Orientation: 0.0010; Line Search: 4.8646
    LBFGS Accumulation History: 1 points
    th(0)=0.3878335255338854;dx=-759.2748912355022
    New Minimum: 0.3878335255338854 > 0.3862960492323624
    END: th(5.7951313856138046E-6)=0.3862960492323624; dx=-301.6715763921537 delta=0.0015374763015230108
    Iteration 42 complete. Error: 0.3862960492323624 Total: 15340974313763.9470; Orientation: 0.0007; Line Search: 1.9436
    LBFGS Accumulation History: 1 points
    th(0)=0.3862960492323624;dx=-484.7655144208693
    New Minimum: 0.3862960492323624 > 0.3838981649484545
    END: th(1.2485232090458919E-5)=0.3838981649484545; dx=-284.4955895287655 delta=0.0023978842839078585
    Iteration 43 complete. Error: 0.3838981649484545 Total: 15343917364722.0250; Orientation: 0.0007; Line Search: 1.9644
    LBFGS Accumulation History: 1 points
    th(0)=0.3838981649484545;dx=-703.6513441480366
    Armijo: th(2.689861712878399E-5)=0.3884368578387941; dx=1377.8102844180298 delta=-0.004538692890339591
    New Minimum: 0.3838981649484545 > 0.3826706987996883
    WOLF (strong): th(1.3449308564391995E-5)=0.3826706987996883; dx=337.38333121521714 delta=0.001227466148766232
    END: th(4.4831028547973315E-6)=0.3827106164197432; dx=-355.3851935321269 delta=0.0011875485287113352
    Iteration 44 complete. Error: 0.3826706987996883 Total: 15348740113557.0820; Orientation: 0.0010; Line Search: 3.8584
    
```

Returns: 

```
    0.3826706987996883
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
      "FullyConnectedLayer/3cc8990a-29bd-4377-9ee9-863800000083" : [ "java.util.HashMap", {
        "avgMsPerItem" : 0.007915022337634409,
        "medianMsPerItem" : "NaN",
        "avgMsPerItem_Backward" : 5.912321110107529E-6,
        "totalItems" : 1860000,
        "backpropStatistics" : [ "java.util.HashMap", {
          "meanExponent" : -6.383490560244054,
          "tp50" : -2.04288184087452E-5,
          "negative" : 5000,
          "min" : -1.9983022942938116E-4,
          "max" : 1.89634018708417E-4,
          "tp90" : -1.817340717161606E-6,
          "mean" : 3.1163536533053795E-23,
          "count" : 50000.0,
          "positive" : 45000,
          "stdDev" : 2.588796888919616E-5,
          "tp75" : -6.151694503904379E-6,
          "zeros" : 0
        } ],
        "totalBatches" : 372,
        "weights" : [ "java.util.HashMap", {
          "tp50" : "NaN",
          "buffers" : 1,
          "max" : 0.0018873212837172847,
          "tp90" : "NaN",
          "count" : 7840.0,
          "positive" : 4307,
          "tp75" : "NaN",
          "zeros" : 0,
          "meanExponent" : -3.6745478856534723,
          "negative" : 3533,
          "min" : -0.0016851246818492193,
          "mean" : 5.2534526593741515E-5,
          "stdDev" : 3.8840065380965286E-4
        } ],
        "class" : "com.simiacryptus.mindseye.layers.java.FullyConnectedLayer",
        "outputStatistics" : [ "java.util.HashMap", {
          "meanExponent" : 0.20572836224761973,
          "tp50" : -3.037499292272226,
          "negative" : 16106,
          "min" : -5.642430188839287,
          "max" : 11.005277120792028,
          "tp90" : -2.276946992921061,
          "mean" : 1.4835041386403836,
          "count" : 50000.0,
          "positive" : 33894,
          "stdDev" : 3.0695373639345584,
          "tp75" : -2.5226376391593424,
          "zeros" : 0
        } ],
        "medianMsPerItem_Backward" : "NaN"
      } ],
      "BiasLayer/3cc8990a-29bd-4377-9ee9-863800000082" : [ "java.util.HashMap", {
        "avgMsPerItem" : 0.012373391387096784,
        "medianMsPerItem" : "NaN",
        "avgMsPerItem_Backward" : 2.956607436021505E-6,
        "totalItems" : 1860000,
        "backpro
```
...[skipping 789 bytes](etc/2.txt)...
```
     "negative" : 404,
          "min" : -8.476617103205908E-9,
          "mean" : -1.43803350281864E-10,
          "stdDev" : 2.5908297884767183E-9
        } ],
        "class" : "com.simiacryptus.mindseye.layers.java.BiasLayer",
        "outputStatistics" : [ "java.util.HashMap", {
          "meanExponent" : -6.7667609042406625,
          "tp50" : -8.476617103205908E-9,
          "negative" : 1617937,
          "min" : -9.51308075989547E-10,
          "max" : -9.51308075989547E-10,
          "tp90" : -8.05300614443699E-9,
          "mean" : 33.2845221937302,
          "count" : 3920000.0,
          "positive" : 2302063,
          "stdDev" : 78.55932605110644,
          "tp75" : -8.05300614443699E-9,
          "zeros" : 0
        } ],
        "medianMsPerItem_Backward" : "NaN"
      } ],
      "SoftmaxActivationLayer/3cc8990a-29bd-4377-9ee9-863800000084" : [ "java.util.HashMap", {
        "avgMsPerItem" : 0.0019660602333333344,
        "medianMsPerItem" : "NaN",
        "avgMsPerItem_Backward" : 9.025060344086018E-8,
        "totalItems" : 1860000,
        "backpropStatistics" : [ "java.util.HashMap", {
          "meanExponent" : -3.533777468878853,
          "tp50" : -2.2275289189247066E-4,
          "negative" : 5000,
          "min" : -0.23561209610233577,
          "max" : 0.0,
          "tp90" : -2.0183400578409634E-4,
          "mean" : -9.860709079730022E-5,
          "count" : 50000.0,
          "positive" : 0,
          "stdDev" : 0.00568795756999436,
          "tp75" : -2.063469159435374E-4,
          "zeros" : 45000
        } ],
        "totalBatches" : 372,
        "class" : "com.simiacryptus.mindseye.layers.java.SoftmaxActivationLayer",
        "outputStatistics" : [ "java.util.HashMap", {
          "meanExponent" : -2.5938016718581483,
          "tp50" : 6.297402423232049E-6,
          "negative" : 0,
          "min" : 1.6255686467624268E-8,
          "max" : 0.9917283054222177,
          "tp90" : 2.269574799241751E-5,
          "mean" : 0.1,
          "count" : 50000.0,
          "positive" : 50000,
          "stdDev" : 0.2519861868862243,
          "tp75" : 1.4913342461223583E-5,
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
    89.41
```



Let's examine some incorrectly predicted results in more detail:

Code from [MnistTestBase.java:218](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/mnist/MnistTestBase.java#L218) executed in 0.22 seconds: 
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
![[5]](etc/test.2.png)  | 6 (75.8%), 2 (11.8%), 0 (4.7%) 
![[4]](etc/test.3.png)  | 0 (45.2%), 6 (33.4%), 2 (8.2%) 
![[1]](etc/test.4.png)  | 3 (48.9%), 1 (23.8%), 8 (8.1%) 
![[3]](etc/test.5.png)  | 2 (63.1%), 3 (26.6%), 9 (3.2%) 
![[2]](etc/test.6.png)  | 7 (82.6%), 2 (9.0%), 9 (6.5%)  
![[9]](etc/test.7.png)  | 4 (29.8%), 9 (27.1%), 8 (22.9%)
![[7]](etc/test.8.png)  | 1 (48.6%), 7 (26.5%), 9 (10.4%)
![[7]](etc/test.9.png)  | 4 (54.5%), 9 (27.0%), 7 (11.7%)
![[2]](etc/test.10.png) | 9 (50.4%), 8 (12.9%), 4 (11.5%)
![[9]](etc/test.11.png) | 3 (29.2%), 4 (28.5%), 9 (27.9%)




