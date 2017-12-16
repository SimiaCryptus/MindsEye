# SignReducerLayer
## Normal
### Network Diagram
This is a network with the following layout:

Code from [StandardLayerTests.java:72](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/StandardLayerTests.java#L72) executed in 0.31 seconds: 
```java
    return Graphviz.fromGraph(TestUtil.toGraph((DAGNetwork) layer))
      .height(400).width(600).render(Format.PNG).toImage();
```

Returns: 

![Result](etc/test.233.png)



### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.00 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (40#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (40#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.01 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [ 1.704, 0.676, -1.836 ]
    Inputs Statistics: {meanExponent=0.10842965441251362, negative=1, min=-1.836, max=-1.836, mean=0.18133333333333326, count=3.0, positive=2, stdDev=1.4869255379996524, zeros=0}
    Output: [ 0.5304502354229002 ]
    Outputs Statistics: {meanExponent=-0.2753553535062643, negative=0, min=0.5304502354229002, max=0.5304502354229002, mean=0.5304502354229002, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Feedback for input 0
    Inputs Values: [ 1.704, 0.676, -1.836 ]
    Value Statistics: {meanExponent=0.10842965441251362, negative=1, min=-1.836, max=-1.836, mean=0.18133333333333326, count=3.0, positive=2, stdDev=1.4869255379996524, zeros=0}
    Implemented Feedback: [ [ 0.04886318997256545 ], [ 0.05357088279062453 ], [ 0.06507450570790507 ] ]
    Implemented Statistics: {meanExponent=-1.2562261676460773, negative=0, min=0.06507450570790507, max=0.06507450570790507, mean=0.05583619282369836, count=3.0, positive=3, stdDev=0.00680932750605986, zeros=0}
    Measured Feedback: [ [ 0.04886196646292085 ], [ 0.
```
...[skipping 650 bytes](etc/329.txt)...
```
    87155995543631E-4 ], [ -0.006869201582771027 ] ]
    Implemented Statistics: {meanExponent=-2.9046160937039325, negative=2, min=-0.006869201582771027, max=-0.006869201582771027, mean=-0.0035475365713632315, count=2.0, positive=0, stdDev=0.0033216650114077954, zeros=0}
    Measured Gradient: [ [ -2.2587130943385603E-4 ], [ -0.0068689691501422345 ] ]
    Measured Statistics: {meanExponent=-2.904623682267488, negative=2, min=-0.0068689691501422345, max=-0.0068689691501422345, mean=-0.0035474202297880453, count=2.0, positive=0, stdDev=0.0033215489203541892, zeros=0}
    Gradient Error: [ [ 2.505215802872402E-10 ], [ 2.3243262879234922E-7 ] ]
    Error Statistics: {meanExponent=-8.117428881683622, negative=0, min=2.3243262879234922E-7, max=2.3243262879234922E-7, mean=1.1634157518631823E-7, count=2.0, positive=2, stdDev=1.1609105360603099E-7, zeros=0}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 7.9007e-07 +- 6.9763e-07 [2.5052e-10 - 1.9153e-06] (5#)
    relativeTol: 1.0022e-05 +- 6.1125e-06 [5.5457e-07 - 1.6919e-05] (5#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=7.9007e-07 +- 6.9763e-07 [2.5052e-10 - 1.9153e-06] (5#), relativeTol=1.0022e-05 +- 6.1125e-06 [5.5457e-07 - 1.6919e-05] (5#)}
```



### Json Serialization
Code from [JsonTest.java:36](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/JsonTest.java#L36) executed in 0.00 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.java.SignReducerLayer",
      "id": "8d361cb8-24ba-4638-8a79-56c61763f5ae",
      "isFrozen": false,
      "name": "SignReducerLayer/8d361cb8-24ba-4638-8a79-56c61763f5ae",
      "inputs": [
        "d1bdf82f-cb20-4669-b9b6-fc5e3de0edbf"
      ],
      "nodes": {
        "d35a5587-9d14-462a-ba20-fd417ee0b4e2": "6d5b8dea-c3df-4fcb-ab3d-600bd2018eea",
        "0ede591d-a006-452c-832e-fe86a758df4e": "15547356-89ad-407d-8920-de691d39e5d4",
        "cd378654-2c6d-47f8-a661-e732d3348c94": "28da4fc7-980f-4d70-a19d-a3a34609b027",
        "8a376e27-ce1c-4ef8-86e2-900591f9e2c7": "8aa0affb-7c73-42bf-ab65-35fdff90931e",
        "3dbb639a-9d48-4be9-adf3-a9eeba41148b": "4a6f7b81-1d46-4435-8372-613a09bca236",
        "21aff775-0096-4d8d-845a-672fc398be25": "933dcfbe-6bac-47ed-9ecf-d95f35a49ec6",
        "f1dbc2b9-62b9-4cd0-95a9-357c145f18bc": "82a83719-f15a-4a47-a3f1-c75151697def",
        "0e43de7b-6d39-4686-9fe6-dce8a3bba2fb": "1295767c-2bbf-4610-9d9c-1300e22e91b7",
        "0ba9a7c0-9bfa-47ff-990a-7c9803b37e70": "e9ba4bc4-fc81-4a2e-adb9-e
```
...[skipping 3148 bytes](etc/330.txt)...
```
    52c-832e-fe86a758df4e": [
          "d1bdf82f-cb20-4669-b9b6-fc5e3de0edbf"
        ],
        "cd378654-2c6d-47f8-a661-e732d3348c94": [
          "0ede591d-a006-452c-832e-fe86a758df4e"
        ],
        "8a376e27-ce1c-4ef8-86e2-900591f9e2c7": [
          "d35a5587-9d14-462a-ba20-fd417ee0b4e2"
        ],
        "3dbb639a-9d48-4be9-adf3-a9eeba41148b": [
          "8a376e27-ce1c-4ef8-86e2-900591f9e2c7"
        ],
        "21aff775-0096-4d8d-845a-672fc398be25": [
          "cd378654-2c6d-47f8-a661-e732d3348c94",
          "3dbb639a-9d48-4be9-adf3-a9eeba41148b"
        ],
        "f1dbc2b9-62b9-4cd0-95a9-357c145f18bc": [
          "21aff775-0096-4d8d-845a-672fc398be25"
        ],
        "0e43de7b-6d39-4686-9fe6-dce8a3bba2fb": [
          "f1dbc2b9-62b9-4cd0-95a9-357c145f18bc"
        ],
        "0ba9a7c0-9bfa-47ff-990a-7c9803b37e70": [
          "d35a5587-9d14-462a-ba20-fd417ee0b4e2",
          "0e43de7b-6d39-4686-9fe6-dce8a3bba2fb"
        ],
        "0321af4a-8dd8-4dd6-be40-0264541eb38d": [
          "0ba9a7c0-9bfa-47ff-990a-7c9803b37e70"
        ]
      },
      "labels": {},
      "head": "0321af4a-8dd8-4dd6-be40-0264541eb38d"
    }
```



### Example Input/Output Pair
Code from [ReferenceIO.java:68](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/ReferenceIO.java#L68) executed in 0.00 seconds: 
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
    [[ -0.08, 0.824, 0.208 ]]
    --------------------
    Output: 
    [ 0.6987985759516199 ]
    --------------------
    Derivative: 
    [ 0.35107407634102444, -0.024344858415059623, 0.2314715838523605 ]
```



### Input Learning
In this test, we use a network to learn this target input, given it's pre-evaluated output:

Code from [LearningTester.java:127](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L127) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_target).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [ 0.412, 0.944, -0.956 ]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 0.00 seconds: 
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



This training run resulted in the following regressed input:

Code from [LearningTester.java:144](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L144) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_gd).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [ 0.412, -0.956, 0.944 ]
```



Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 0.00 seconds: 
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



This training run resulted in the following regressed input:

Code from [LearningTester.java:154](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L154) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_lbgfs).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [ 0.412, -0.956, 0.944 ]
```



Code from [LearningTester.java:95](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.00 seconds: 
```java
    return TestUtil.compare(runs);
```

Code from [LearningTester.java:98](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

### Model Learning
In this test, attempt to train a network to emulate a randomized network given an example input/output. The target state is:

Code from [LearningTester.java:176](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L176) executed in 0.00 seconds: 
```java
    return network_target.state().stream().map(Arrays::toString).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [-1.0, 0.0]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 0.00 seconds: 
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
    Zero gradient: 0.0
    Constructing line search parameters: GD
    F(0.0) = LineSearchPoint{point=PointSample{avg=0.0017267777721666642}, derivative=0.0}
    Iteration 1 failed, aborting. Error: 0.0017267777721666642 Total: 239719606631165.4000; Orientation: 0.0000; Line Search: 0.0001
    
```

Returns: 

```
    0.0017267777721666642
```



This training run resulted in the following configuration:

Code from [LearningTester.java:189](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L189) executed in 0.00 seconds: 
```java
    return network_gd.state().stream().map(Arrays::toString).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [0.0, -1.0]
    [0.5415545156651677]
```



Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 0.00 seconds: 
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

Code from [LearningTester.java:203](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L203) executed in 0.00 seconds: 
```java
    return network_lbfgs.state().stream().map(Arrays::toString).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [-1.0, 0.0]
    [0.5415545156651677]
```



Code from [LearningTester.java:95](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.00 seconds: 
```java
    return TestUtil.compare(runs);
```

Code from [LearningTester.java:98](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

### Performance
Adding performance wrappers

Code from [TestUtil.java:287](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/TestUtil.java#L287) executed in 0.00 seconds: 
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

Code from [PerformanceTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/PerformanceTester.java#L66) executed in 0.02 seconds: 
```java
    test(component, inputPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[3]
    Performance:
    	Evaluation performance: 0.001366s +- 0.000139s [0.001211s - 0.001620s]
    	Learning performance: 0.000599s +- 0.000070s [0.000536s - 0.000736s]
    
```

Per-layer Performance Metrics:

Code from [TestUtil.java:252](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/TestUtil.java#L252) executed in 0.00 seconds: 
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
    	Optional[SumInputsLayer/933dcfbe-6bac-47ed-9ecf-d95f35a49ec6 -> 0.000126s +- 0.000058s (11.0)
    	SqActivationLayer/8aa0affb-7c73-42bf-ab65-35fdff90931e -> 0.000136s +- 0.000062s (11.0)
    	AvgReducerLayer/28da4fc7-980f-4d70-a19d-a3a34609b027 -> 0.000087s +- 0.000031s (11.0)
    	NthPowerActivationLayer/1295767c-2bbf-4610-9d9c-1300e22e91b7 -> 0.000119s +- 0.000037s (11.0)
    	SigmoidActivationLayer/fc3dbe06-5eca-4115-aeb6-64ccd653a17e -> 0.000122s +- 0.000038s (11.0)
    	ProductInputsLayer/e9ba4bc4-fc81-4a2e-adb9-e2257b1bddf2 -> 0.000086s +- 0.000027s (11.0)
    	SqActivationLayer/15547356-89ad-407d-8920-de691d39e5d4 -> 0.000128s +- 0.000043s (11.0)
    	AvgReducerLayer/6d5b8dea-c3df-4fcb-ab3d-600bd2018eea -> 0.000096s +- 0.000026s (11.0)
    	NthPowerActivationLayer/82a83719-f15a-4a47-a3f1-c75151697def -> 0.000127s +- 0.000040s (11.0)
    	LinearActivationLayer/4a6f7b81-1d46-4435-8372-613a09bca236 -> 0.000041s +- 0.000015s (11.0)]
    Backward Performance: 
    	Optional[SumInputsLayer/933dcfbe-6bac-47ed-9ecf-d95f35a49ec6 -> 0.000005s +- 0.000010s (6.0)
    	SqActivationLayer/8aa0affb-7c73-42bf-ab65-35fdff90931e -> 0.000001s +- 0.000000s (1.0)
    	AvgReducerLayer/28da4fc7-980f-4d70-a19d-a3a34609b027 -> 0.000008s +- 0.000000s (1.0)
    	NthPowerActivationLayer/1295767c-2bbf-4610-9d9c-1300e22e91b7 -> 0.000004s +- 0.000007s (6.0)
    	SigmoidActivationLayer/fc3dbe06-5eca-4115-aeb6-64ccd653a17e -> 0.000006s +- 0.000011s (6.0)
    	ProductInputsLayer/e9ba4bc4-fc81-4a2e-adb9-e2257b1bddf2 -> 0.000012s +- 0.000024s (6.0)
    	SqActivationLayer/15547356-89ad-407d-8920-de691d39e5d4 -> 0.000002s +- 0.000000s (1.0)
    	AvgReducerLayer/6d5b8dea-c3df-4fcb-ab3d-600bd2018eea -> 0.000003s +- 0.000000s (1.0)
    	NthPowerActivationLayer/82a83719-f15a-4a47-a3f1-c75151697def -> 0.000003s +- 0.000004s (6.0)
    	LinearActivationLayer/4a6f7b81-1d46-4435-8372-613a09bca236 -> 0.000003s +- 0.000006s (6.0)]
    
```

Removing performance wrappers

Code from [TestUtil.java:270](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/TestUtil.java#L270) executed in 0.00 seconds: 
```java
    network.visitNodes(node -> {
      if (node.getLayer() instanceof MonitoringWrapperLayer) {
        node.setLayer(node.<MonitoringWrapperLayer>getLayer().getInner());
      }
    });
```

