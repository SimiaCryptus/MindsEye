# StdDevMetaLayer
## StdDevMetaLayerTest
### Network Diagram
This is a network with the following layout:

Code from [StandardLayerTests.java:72](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/StandardLayerTests.java#L72) executed in 0.22 seconds: 
```java
    return Graphviz.fromGraph(TestUtil.toGraph((DAGNetwork) layer))
      .height(400).width(600).render(Format.PNG).toImage();
```

Returns: 

![Result](etc/test.242.png)



### Json Serialization
Code from [JsonTest.java:36](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/JsonTest.java#L36) executed in 0.00 seconds: 
```java
    JsonObject json = layer.getJson();
    NNLayer echo = NNLayer.fromJson(json);
    if ((echo == null)) throw new AssertionError("Failed to deserialize");
    if ((layer == echo)) throw new AssertionError("Serialization did not copy");
    if ((!layer.equals(echo))) throw new AssertionError("Serialization not equal");
    return new GsonBuilder().setPrettyPrinting().create().toJson(json);
```

Returns: 

```
    {
      "class": "com.simiacryptus.mindseye.layers.java.StdDevMetaLayer",
      "id": "c5955b21-18be-4c58-9dd6-70d755beb475",
      "isFrozen": false,
      "name": "StdDevMetaLayer/c5955b21-18be-4c58-9dd6-70d755beb475",
      "inputs": [
        "fa8c8b64-8826-4e97-8c8d-ec2538799859"
      ],
      "nodes": {
        "fd2c2441-052d-4373-897d-d168b7ad76a5": "bf7d5e03-0d15-4baf-b129-8a28e0905664",
        "da4b0275-03d4-47de-86fb-4492ae733e01": "c842a6e2-0c4d-4eb3-995e-2430252a63ec",
        "455088f6-7730-404b-94d0-2d8ef1a54e85": "227dd345-06e8-4b56-a056-98395068ce1d",
        "2199e281-a207-4ee5-b6fa-fa0537ea23b9": "4c2d26a6-bad9-40af-8607-c0ac5baffae5",
        "adee4976-ddf3-42d1-8dba-fd8e14cf16b8": "bb198ce8-2701-4019-821d-352a334fb9a8",
        "403745c4-8a0a-4cd5-aedc-c30ea85ec588": "08f7cf16-14de-48cc-be6a-409c14e3bc0b",
        "c62bee80-3d3a-4e44-a70e-c83532c10d4f": "a9301f54-1d49-4965-98e2-cc70eb523b2a",
        "92507471-f430-4706-b5ca-f169ca6d6bd0": "380c8e5c-471f-4124-9235-5764c4bfa453",
        "03bdf397-90ca-4fc8-a6a0-b95ae6af47ff": "f634dc5e-e9ef-4eb9-9d3b-0f4
```
...[skipping 2605 bytes](etc/403.txt)...
```
    5
        }
      },
      "links": {
        "fd2c2441-052d-4373-897d-d168b7ad76a5": [
          "fa8c8b64-8826-4e97-8c8d-ec2538799859"
        ],
        "da4b0275-03d4-47de-86fb-4492ae733e01": [
          "fd2c2441-052d-4373-897d-d168b7ad76a5"
        ],
        "455088f6-7730-404b-94d0-2d8ef1a54e85": [
          "da4b0275-03d4-47de-86fb-4492ae733e01"
        ],
        "2199e281-a207-4ee5-b6fa-fa0537ea23b9": [
          "fa8c8b64-8826-4e97-8c8d-ec2538799859"
        ],
        "adee4976-ddf3-42d1-8dba-fd8e14cf16b8": [
          "2199e281-a207-4ee5-b6fa-fa0537ea23b9"
        ],
        "403745c4-8a0a-4cd5-aedc-c30ea85ec588": [
          "adee4976-ddf3-42d1-8dba-fd8e14cf16b8"
        ],
        "c62bee80-3d3a-4e44-a70e-c83532c10d4f": [
          "403745c4-8a0a-4cd5-aedc-c30ea85ec588"
        ],
        "92507471-f430-4706-b5ca-f169ca6d6bd0": [
          "455088f6-7730-404b-94d0-2d8ef1a54e85",
          "c62bee80-3d3a-4e44-a70e-c83532c10d4f"
        ],
        "03bdf397-90ca-4fc8-a6a0-b95ae6af47ff": [
          "92507471-f430-4706-b5ca-f169ca6d6bd0"
        ]
      },
      "labels": {},
      "head": "03bdf397-90ca-4fc8-a6a0-b95ae6af47ff"
    }
```



### Example Input/Output Pair
Code from [ReferenceIO.java:68](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/ReferenceIO.java#L68) executed in 0.00 seconds: 
```java
    SimpleEval eval = SimpleEval.run(layer, inputPrototype);
    return String.format("--------------------\nInput: \n[%s]\n--------------------\nOutput: \n%s\n--------------------\nDerivative: \n%s",
      Arrays.stream(inputPrototype).map(t -> t.prettyPrint()).reduce((a, b) -> a + ",\n" + b).get(),
      eval.getOutput().prettyPrint(),
      Arrays.stream(eval.getDerivative()).map(t -> t.prettyPrint()).reduce((a, b) -> a + ",\n" + b).get());
```

Returns: 

```
    --------------------
    Input: 
    [[ 0.692, -0.78, 1.184 ]]
    --------------------
    Output: 
    [ 0.8344089058862101 ]
    --------------------
    Derivative: 
    [ 0.1304982342838731, -0.45754278877488574, 0.32704455449101266 ]
```



### Differential Validation
Code from [BatchDerivativeTester.java:76](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchDerivativeTester.java#L76) executed in 0.02 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [ -0.304, 0.376, -0.98 ],
    [ -0.304, 0.376, -0.98 ],
    [ -0.304, 0.376, -0.98 ],
    [ -0.304, 0.376, -0.98 ],
    [ -0.304, 0.376, -0.98 ],
    [ -0.304, 0.376, -0.98 ],
    [ -0.304, 0.376, -0.98 ],
    [ -0.304, 0.376, -0.98 ],
    [ -0.304, 0.376, -0.98 ],
    [ -0.304, 0.376, -0.98 ]
    Inputs Statistics: {meanExponent=-0.3169041652570301, negative=2, min=-0.98, max=-0.98, mean=-0.30266666666666664, count=3.0, positive=1, stdDev=0.5535854847165782, zeros=0},
    {meanExponent=-0.3169041652570301, negative=2, min=-0.98, max=-0.98, mean=-0.30266666666666664, count=3.0, positive=1, stdDev=0.5535854847165782, zeros=0},
    {meanExponent=-0.3169041652570301, negative=2, min=-0.98, max=-0.98, mean=-0.30266666666666664, count=3.0, positive=1, stdDev=0.5535854847165782, zeros=0},
    {meanExponent=-0.3169041652570301, negative=2, min=-0.98, max=-0.98, mean=-0.30266666666666664, count=3.0, positive=1, stdDev=0.5535854847165782, zeros=0},
    {meanExponent=-0.3169041652570301, negative=2, min=-0.98, max=-0.98, mean=-0.30266666666666664, count=3.0, positi
```
...[skipping 12793 bytes](etc/404.txt)...
```
    .0 ]
    Implemented Gradient: [ [ 0.08273980590189399 ], [ 0.9032028725536172 ] ]
    Implemented Statistics: {meanExponent=-0.5632500960458364, negative=0, min=0.9032028725536172, max=0.9032028725536172, mean=0.4929713392277556, count=2.0, positive=2, stdDev=0.4102315333258616, zeros=0}
    Measured Gradient: [ [ 0.08273918758905374 ], [ 0.9031292035011074 ] ]
    Measured Statistics: {meanExponent=-0.5632694309599122, negative=0, min=0.9031292035011074, max=0.9031292035011074, mean=0.49293419554508056, count=2.0, positive=2, stdDev=0.4101950079560268, zeros=0}
    Gradient Error: [ [ -6.183128402476035E-7 ], [ -7.366905250982825E-5 ] ]
    Error Statistics: {meanExponent=-5.170753325177618, negative=2, min=-7.366905250982825E-5, max=-7.366905250982825E-5, mean=-3.7143682675037926E-5, count=2.0, positive=0, stdDev=3.652536983479032E-5, zeros=0}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 4.9558e-06 +- 1.2347e-05 [6.1831e-07 - 7.3669e-05] (32#)
    relativeTol: 5.7911e-03 +- 8.5411e-03 [3.7365e-06 - 1.8460e-02] (32#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=4.9558e-06 +- 1.2347e-05 [6.1831e-07 - 7.3669e-05] (32#), relativeTol=5.7911e-03 +- 8.5411e-03 [3.7365e-06 - 1.8460e-02] (32#)}
```



### Performance
Adding performance wrappers

Code from [TestUtil.java:302](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/TestUtil.java#L302) executed in 0.00 seconds: 
```java
    network.visitNodes(node -> {
      if (!(node.getLayer() instanceof MonitoringWrapperLayer)) {
        node.setLayer(new MonitoringWrapperLayer(node.getLayer()).shouldRecordSignalMetrics(false));
      }
      else {
        ((MonitoringWrapperLayer) node.getLayer()).shouldRecordSignalMetrics(false);
      }
    });
```

Now we execute larger-scale runs to benchmark performance:

Code from [PerformanceTester.java:66](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/PerformanceTester.java#L66) executed in 0.01 seconds: 
```java
    test(component, inputPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[100]
    Performance:
    	Evaluation performance: 0.000885s +- 0.000199s [0.000731s - 0.001264s]
    	Learning performance: 0.000015s +- 0.000014s [0.000007s - 0.000042s]
    
```

Per-layer Performance Metrics:

Code from [TestUtil.java:267](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/TestUtil.java#L267) executed in 0.00 seconds: 
```java
    Map<NNLayer, MonitoringWrapperLayer> metrics = new HashMap<>();
    network.visitNodes(node -> {
      if ((node.getLayer() instanceof MonitoringWrapperLayer)) {
        MonitoringWrapperLayer layer = node.getLayer();
        metrics.put(layer.getInner(), layer);
      }
    });
    System.out.println("Forward Performance: \n\t" + metrics.entrySet().stream().map(e -> {
      PercentileStatistics performance = e.getValue().getForwardPerformance();
      return String.format("%s -> %.6fs +- %.6fs (%s)", e.getKey(), performance.getMean(), performance.getStdDev(), performance.getCount());
    }).reduce((a, b) -> a + "\n\t" + b));
    System.out.println("Backward Performance: \n\t" + metrics.entrySet().stream().map(e -> {
      PercentileStatistics performance = e.getValue().getBackwardPerformance();
      return String.format("%s -> %.6fs +- %.6fs (%s)", e.getKey(), performance.getMean(), performance.getStdDev(), performance.getCount());
    }).reduce((a, b) -> a + "\n\t" + b));
```
Logging: 
```
    Forward Performance: 
    	Optional[SumInputsLayer/ccfc66cd-9d4d-475e-95df-3b63ca588e06 -> 0.000012s +- 0.000004s (11.0)
    	SqActivationLayer/b69b7a2d-8340-4288-83f5-69c7ac44cd3a -> 0.000002s +- 0.000002s (11.0)
    	SqActivationLayer/7bdce012-6486-4502-96d0-73e968de04ba -> 0.000129s +- 0.000040s (11.0)
    	AvgMetaLayer/5b31cb4a-fc8e-49a4-828b-7c0d9243b32c -> 0.000290s +- 0.000090s (11.0)
    	AvgReducerLayer/a252d0ae-4d52-4490-882a-7b4f62ea7819 -> 0.000004s +- 0.000002s (11.0)
    	LinearActivationLayer/eb230d56-45e4-4658-8393-2ba04fac3d4f -> 0.000002s +- 0.000001s (11.0)
    	AvgMetaLayer/9f0650e5-a3c6-4cda-be18-7f11e82b3ef1 -> 0.000231s +- 0.000075s (11.0)
    	AvgReducerLayer/87fde7fc-3962-453f-8fd1-4cebeb1cefe6 -> 0.000003s +- 0.000002s (11.0)
    	NthPowerActivationLayer/99f2de3a-ae42-4339-8b1a-cc379fc87289 -> 0.000005s +- 0.000009s (11.0)]
    Backward Performance: 
    	Optional[SumInputsLayer/ccfc66cd-9d4d-475e-95df-3b63ca588e06 -> 0.000011s +- 0.000000s (1.0)
    	SqActivationLayer/b69b7a2d-8340-4288-83f5-69c7ac44cd3a -> 0.000001s +- 0.000000s (1.0)
    	SqActivationLayer/7bdce012-6486-4502-96d0-73e968de04ba -> 0.000004s +- 0.000000s (1.0)
    	AvgMetaLayer/5b31cb4a-fc8e-49a4-828b-7c0d9243b32c -> 0.000029s +- 0.000000s (1.0)
    	AvgReducerLayer/a252d0ae-4d52-4490-882a-7b4f62ea7819 -> 0.000008s +- 0.000000s (1.0)
    	LinearActivationLayer/eb230d56-45e4-4658-8393-2ba04fac3d4f -> 0.000003s +- 0.000000s (1.0)
    	AvgMetaLayer/9f0650e5-a3c6-4cda-be18-7f11e82b3ef1 -> 0.000025s +- 0.000000s (1.0)
    	AvgReducerLayer/87fde7fc-3962-453f-8fd1-4cebeb1cefe6 -> 0.000003s +- 0.000000s (1.0)
    	NthPowerActivationLayer/99f2de3a-ae42-4339-8b1a-cc379fc87289 -> 0.000011s +- 0.000011s (6.0)]
    
```

Removing performance wrappers

Code from [TestUtil.java:285](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/TestUtil.java#L285) executed in 0.00 seconds: 
```java
    network.visitNodes(node -> {
      if (node.getLayer() instanceof MonitoringWrapperLayer) {
        node.setLayer(node.<MonitoringWrapperLayer>getLayer().getInner());
      }
    });
```

### Input Learning
In this test, we use a network to learn this target input, given it's pre-evaluated output:

Code from [LearningTester.java:127](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L127) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_target).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [ -1.444, -1.396, 1.484, -0.148, -0.836, -1.776, 1.364, 1.812, ... ]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:300](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L300) executed in 0.01 seconds: 
```java
    return new IterativeTrainer(trainable)
      .setLineSearchFactory(label -> new QuadraticSearch())
      .setOrientation(new GradientDescent())
      .setMonitor(monitor)
      .setTimeout(30, TimeUnit.SECONDS)
      .setMaxIterations(250)
      .setTerminateThreshold(0)
      .run();
```
Logging: 
```
    Zero gradient: 2.492439167706937E-16
    Constructing line search parameters: GD
    F(0.0) = LineSearchPoint{point=PointSample{avg=4.437342591868191E-31}, derivative=-6.21225300471965E-32}
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=4.437342591868191E-31}, derivative=-6.21225300471965E-32}, delta = 0.0
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=4.437342591868191E-31}, derivative=-6.21225300471965E-32}, delta = 0.0
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=4.437342591868191E-31}, derivative=-6.21225300471965E-32}, delta = 0.0
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=4.437342591868191E-31}, derivative=-6.21225300471965E-32}, delta = 0.0
    F(2.4010000000000004E-7) = LineSearchPoint{point=PointSample{avg=4.437342591868191E-31}, derivative=-6.21225300471965E-32}, delta = 0.0
    F(1.6807000000000003E-6) = LineSearchPoint{point=PointSample{avg=4.437342591868191E-31}, derivative=-6.21225300471965E-32}, delta = 0.0
    F(1.1764900000000001E-5) = LineSe
```
...[skipping 2050 bytes](etc/405.txt)...
```
    .0
    Right bracket at 0.01081350562578125
    F(0.005406752812890625) = LineSearchPoint{point=PointSample{avg=4.437342591868191E-31}, derivative=-6.21225300471965E-32}, delta = 0.0
    Right bracket at 0.005406752812890625
    F(0.0027033764064453127) = LineSearchPoint{point=PointSample{avg=4.437342591868191E-31}, derivative=-6.21225300471965E-32}, delta = 0.0
    Right bracket at 0.0027033764064453127
    F(0.0013516882032226563) = LineSearchPoint{point=PointSample{avg=4.437342591868191E-31}, derivative=-6.21225300471965E-32}, delta = 0.0
    Right bracket at 0.0013516882032226563
    F(6.758441016113282E-4) = LineSearchPoint{point=PointSample{avg=4.437342591868191E-31}, derivative=-6.21225300471965E-32}, delta = 0.0
    Right bracket at 6.758441016113282E-4
    F(3.379220508056641E-4) = LineSearchPoint{point=PointSample{avg=4.437342591868191E-31}, derivative=-6.21225300471965E-32}, delta = 0.0
    Loops = 12
    Iteration 1 failed, aborting. Error: 4.437342591868191E-31 Total: 249862192481877.8000; Orientation: 0.0000; Line Search: 0.0064
    
```

Returns: 

```
    4.437342591868191E-31
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:144](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L144) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_gd).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [ -0.348, 1.792, 1.84, 1.0, -1.876, -1.444, -0.712, -1.988, ... ]
```



Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:324](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L324) executed in 0.00 seconds: 
```java
    return new IterativeTrainer(trainable)
      .setLineSearchFactory(label -> new ArmijoWolfeSearch())
      .setOrientation(new LBFGS())
      .setMonitor(monitor)
      .setTimeout(30, TimeUnit.SECONDS)
      .setMaxIterations(250)
      .setTerminateThreshold(0)
      .run();
```
Logging: 
```
    LBFGS Accumulation History: 1 points
    Constructing line search parameters: GD
    th(0)=4.437342591868191E-31;dx=-6.21225300471965E-32
    Armijo: th(2.154434690031884)=4.437342591868191E-31; dx=-6.21225300471965E-32 delta=0.0
    Armijo: th(1.077217345015942)=4.437342591868191E-31; dx=-6.21225300471965E-32 delta=0.0
    Armijo: th(0.3590724483386473)=4.437342591868191E-31; dx=-6.21225300471965E-32 delta=0.0
    Armijo: th(0.08976811208466183)=4.437342591868191E-31; dx=-6.21225300471965E-32 delta=0.0
    Armijo: th(0.017953622416932366)=4.437342591868191E-31; dx=-6.21225300471965E-32 delta=0.0
    Armijo: th(0.002992270402822061)=4.437342591868191E-31; dx=-6.21225300471965E-32 delta=0.0
    Armijo: th(4.2746720040315154E-4)=4.437342591868191E-31; dx=-6.21225300471965E-32 delta=0.0
    Armijo: th(5.343340005039394E-5)=4.437342591868191E-31; dx=-6.21225300471965E-32 delta=0.0
    Armijo: th(5.9370444500437714E-6)=4.437342591868191E-31; dx=-6.21225300471965E-32 delta=0.0
    Armijo: th(5.937044450043771E-7)=4.437342591868191E-31; dx=-6.21225300
```
...[skipping 428 bytes](etc/406.txt)...
```
    2E-9)=4.437342591868191E-31; dx=-6.21225300471965E-32 delta=0.0
    Armijo: th(8.649540282697801E-10)=4.437342591868191E-31; dx=-6.21225300471965E-32 delta=0.0
    WOLFE (weak): th(6.05467819788846E-10)=4.437342591868191E-31; dx=-6.21225300471965E-32 delta=0.0
    Armijo: th(7.35210924029313E-10)=4.437342591868191E-31; dx=-6.21225300471965E-32 delta=0.0
    WOLFE (weak): th(6.703393719090796E-10)=4.437342591868191E-31; dx=-6.21225300471965E-32 delta=0.0
    WOLFE (weak): th(7.027751479691964E-10)=4.437342591868191E-31; dx=-6.21225300471965E-32 delta=0.0
    Armijo: th(7.189930359992546E-10)=4.437342591868191E-31; dx=-6.21225300471965E-32 delta=0.0
    Armijo: th(7.108840919842255E-10)=4.437342591868191E-31; dx=-6.21225300471965E-32 delta=0.0
    Armijo: th(7.068296199767109E-10)=4.437342591868191E-31; dx=-6.21225300471965E-32 delta=0.0
    mu /= nu: th(0)=4.437342591868191E-31;th'(0)=-6.21225300471965E-32;
    Iteration 1 failed, aborting. Error: 4.437342591868191E-31 Total: 249862200445028.8000; Orientation: 0.0001; Line Search: 0.0037
    
```

Returns: 

```
    4.437342591868191E-31
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:154](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L154) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_lbgfs).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [ -0.348, 1.792, 1.84, 1.0, -1.876, -1.444, -0.712, -1.988, ... ]
```



Code from [LearningTester.java:96](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L96) executed in 0.00 seconds: 
```java
    return TestUtil.compare(runs);
```

Code from [LearningTester.java:99](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L99) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

### Model Learning
In this test, attempt to train a network to emulate a randomized network given an example input/output. The target state is:

Code from [LearningTester.java:176](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L176) executed in 0.00 seconds: 
```java
    return network_target.state().stream().map(Arrays::toString).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [0.0, -1.0]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:300](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L300) executed in 0.00 seconds: 
```java
    return new IterativeTrainer(trainable)
      .setLineSearchFactory(label -> new QuadraticSearch())
      .setOrientation(new GradientDescent())
      .setMonitor(monitor)
      .setTimeout(30, TimeUnit.SECONDS)
      .setMaxIterations(250)
      .setTerminateThreshold(0)
      .run();
```

Returns: 

```
    0.0
```



This training run resulted in the following configuration:

Code from [LearningTester.java:189](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L189) executed in 0.00 seconds: 
```java
    return network_gd.state().stream().map(Arrays::toString).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [0.6324574293974264]
    [0.0, -1.0]
```



Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:324](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L324) executed in 0.00 seconds: 
```java
    return new IterativeTrainer(trainable)
      .setLineSearchFactory(label -> new ArmijoWolfeSearch())
      .setOrientation(new LBFGS())
      .setMonitor(monitor)
      .setTimeout(30, TimeUnit.SECONDS)
      .setMaxIterations(250)
      .setTerminateThreshold(0)
      .run();
```

Returns: 

```
    0.0
```



This training run resulted in the following configuration:

Code from [LearningTester.java:203](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L203) executed in 0.00 seconds: 
```java
    return network_lbfgs.state().stream().map(Arrays::toString).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [0.6324574293974264]
    [0.0, -1.0]
```



Code from [LearningTester.java:96](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L96) executed in 0.00 seconds: 
```java
    return TestUtil.compare(runs);
```

Code from [LearningTester.java:99](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L99) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

### Composite Learning
In this test, attempt to train a network to emulate a randomized network given an example input/output. The target state is:

Code from [LearningTester.java:219](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L219) executed in 0.00 seconds: 
```java
    return network_target.state().stream().map(Arrays::toString).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [0.0, -1.0]
```



We simultaneously regress this target input:

Code from [LearningTester.java:223](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L223) executed in 0.00 seconds: 
```java
    return Arrays.stream(testInput).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [ 1.628, -1.116, 0.236, -1.032, 1.368, -1.032, 1.84, -0.724, ... ]
```



Which produces the following output:

Code from [LearningTester.java:230](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L230) executed in 0.00 seconds: 
```java
    return Stream.of(targetOutput).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [ 0.6324574293974261 ]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:300](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L300) executed in 0.01 seconds: 
```java
    return new IterativeTrainer(trainable)
      .setLineSearchFactory(label -> new QuadraticSearch())
      .setOrientation(new GradientDescent())
      .setMonitor(monitor)
      .setTimeout(30, TimeUnit.SECONDS)
      .setMaxIterations(250)
      .setTerminateThreshold(0)
      .run();
```
Logging: 
```
    Zero gradient: 2.0770326397557838E-16
    Constructing line search parameters: GD
    F(0.0) = LineSearchPoint{point=PointSample{avg=3.0814879110195774E-31}, derivative=-4.31406458661088E-32}
    F(1.0E-10) = LineSearchPoint{point=PointSample{avg=3.0814879110195774E-31}, derivative=-4.31406458661088E-32}, delta = 0.0
    F(7.000000000000001E-10) = LineSearchPoint{point=PointSample{avg=3.0814879110195774E-31}, derivative=-4.31406458661088E-32}, delta = 0.0
    F(4.900000000000001E-9) = LineSearchPoint{point=PointSample{avg=3.0814879110195774E-31}, derivative=-4.31406458661088E-32}, delta = 0.0
    F(3.430000000000001E-8) = LineSearchPoint{point=PointSample{avg=3.0814879110195774E-31}, derivative=-4.31406458661088E-32}, delta = 0.0
    F(2.4010000000000004E-7) = LineSearchPoint{point=PointSample{avg=3.0814879110195774E-31}, derivative=-4.31406458661088E-32}, delta = 0.0
    F(1.6807000000000003E-6) = LineSearchPoint{point=PointSample{avg=3.0814879110195774E-31}, derivative=-4.31406458661088E-32}, delta = 0.0
    F(1.1764900000000001E-5) 
```
...[skipping 2078 bytes](etc/407.txt)...
```
    ght bracket at 0.01081350562578125
    F(0.005406752812890625) = LineSearchPoint{point=PointSample{avg=3.0814879110195774E-31}, derivative=-4.31406458661088E-32}, delta = 0.0
    Right bracket at 0.005406752812890625
    F(0.0027033764064453127) = LineSearchPoint{point=PointSample{avg=3.0814879110195774E-31}, derivative=-4.31406458661088E-32}, delta = 0.0
    Right bracket at 0.0027033764064453127
    F(0.0013516882032226563) = LineSearchPoint{point=PointSample{avg=3.0814879110195774E-31}, derivative=-4.31406458661088E-32}, delta = 0.0
    Right bracket at 0.0013516882032226563
    F(6.758441016113282E-4) = LineSearchPoint{point=PointSample{avg=3.0814879110195774E-31}, derivative=-4.31406458661088E-32}, delta = 0.0
    Right bracket at 6.758441016113282E-4
    F(3.379220508056641E-4) = LineSearchPoint{point=PointSample{avg=3.0814879110195774E-31}, derivative=-4.31406458661088E-32}, delta = 0.0
    Loops = 12
    Iteration 1 failed, aborting. Error: 3.0814879110195774E-31 Total: 249862227590929.7800; Orientation: 0.0000; Line Search: 0.0047
    
```

Returns: 

```
    3.0814879110195774E-31
```



This training run resulted in the following configuration:

Code from [LearningTester.java:245](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L245) executed in 0.00 seconds: 
```java
    return network_gd.state().stream().map(Arrays::toString).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [0.6324574293974261]
    [0.0, -1.0]
```



And regressed input:

Code from [LearningTester.java:249](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_gd).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [ 0.9, 1.14, 1.648, -0.976, 0.564, 1.84, 1.804, 1.592, ... ]
```



Which produces the following output:

Code from [LearningTester.java:256](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L256) executed in 0.00 seconds: 
```java
    return Stream.of(regressedOutput).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [ 3.0814879110195774E-31 ]
```



Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:324](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L324) executed in 0.00 seconds: 
```java
    return new IterativeTrainer(trainable)
      .setLineSearchFactory(label -> new ArmijoWolfeSearch())
      .setOrientation(new LBFGS())
      .setMonitor(monitor)
      .setTimeout(30, TimeUnit.SECONDS)
      .setMaxIterations(250)
      .setTerminateThreshold(0)
      .run();
```
Logging: 
```
    LBFGS Accumulation History: 1 points
    Constructing line search parameters: GD
    th(0)=3.0814879110195774E-31;dx=-4.31406458661088E-32
    Armijo: th(2.154434690031884)=3.0814879110195774E-31; dx=-4.31406458661088E-32 delta=0.0
    Armijo: th(1.077217345015942)=3.0814879110195774E-31; dx=-4.31406458661088E-32 delta=0.0
    Armijo: th(0.3590724483386473)=3.0814879110195774E-31; dx=-4.31406458661088E-32 delta=0.0
    Armijo: th(0.08976811208466183)=3.0814879110195774E-31; dx=-4.31406458661088E-32 delta=0.0
    Armijo: th(0.017953622416932366)=3.0814879110195774E-31; dx=-4.31406458661088E-32 delta=0.0
    Armijo: th(0.002992270402822061)=3.0814879110195774E-31; dx=-4.31406458661088E-32 delta=0.0
    Armijo: th(4.2746720040315154E-4)=3.0814879110195774E-31; dx=-4.31406458661088E-32 delta=0.0
    Armijo: th(5.343340005039394E-5)=3.0814879110195774E-31; dx=-4.31406458661088E-32 delta=0.0
    Armijo: th(5.9370444500437714E-6)=3.0814879110195774E-31; dx=-4.31406458661088E-32 delta=0.0
    Armijo: th(5.937044450043771E-7)=3.0814879110195774E-31; dx=
```
...[skipping 461 bytes](etc/408.txt)...
```
    195774E-31; dx=-4.31406458661088E-32 delta=0.0
    Armijo: th(8.649540282697801E-10)=3.0814879110195774E-31; dx=-4.31406458661088E-32 delta=0.0
    Armijo: th(6.05467819788846E-10)=3.0814879110195774E-31; dx=-4.31406458661088E-32 delta=0.0
    WOLFE (weak): th(4.757247155483791E-10)=3.0814879110195774E-31; dx=-4.31406458661088E-32 delta=0.0
    Armijo: th(5.405962676686125E-10)=3.0814879110195774E-31; dx=-4.31406458661088E-32 delta=0.0
    Armijo: th(5.081604916084959E-10)=3.0814879110195774E-31; dx=-4.31406458661088E-32 delta=0.0
    WOLFE (weak): th(4.919426035784375E-10)=3.0814879110195774E-31; dx=-4.31406458661088E-32 delta=0.0
    WOLFE (weak): th(5.000515475934667E-10)=3.0814879110195774E-31; dx=-4.31406458661088E-32 delta=0.0
    WOLFE (weak): th(5.041060196009813E-10)=3.0814879110195774E-31; dx=-4.31406458661088E-32 delta=0.0
    mu /= nu: th(0)=3.0814879110195774E-31;th'(0)=-4.31406458661088E-32;
    Iteration 1 failed, aborting. Error: 3.0814879110195774E-31 Total: 249862238045647.7800; Orientation: 0.0001; Line Search: 0.0038
    
```

Returns: 

```
    3.0814879110195774E-31
```



This training run resulted in the following configuration:

Code from [LearningTester.java:266](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L266) executed in 0.00 seconds: 
```java
    return network_lbfgs.state().stream().map(Arrays::toString).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [0.6324574293974261]
    [0.0, -1.0]
```



And regressed input:

Code from [LearningTester.java:270](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L270) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_lbgfs).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [ 0.9, 1.14, 1.648, -0.976, 0.564, 1.84, 1.804, 1.592, ... ]
```



Which produces the following output:

Code from [LearningTester.java:277](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L277) executed in 0.00 seconds: 
```java
    return Stream.of(regressedOutput).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [ 3.0814879110195774E-31 ]
```



Code from [LearningTester.java:96](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L96) executed in 0.00 seconds: 
```java
    return TestUtil.compare(runs);
```

Code from [LearningTester.java:99](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L99) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

