# MeanSqLossLayer
## MeanSqLossLayerTest
### Network Diagram
This is a network with the following layout:

Code from [StandardLayerTests.java:72](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/StandardLayerTests.java#L72) executed in 0.22 seconds: 
```java
    return Graphviz.fromGraph(TestUtil.toGraph((DAGNetwork) layer))
      .height(400).width(600).render(Format.PNG).toImage();
```

Returns: 

![Result](etc/test.77.png)



### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.03 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (190#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (190#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.03 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -0.344 ], [ -1.072 ], [ 1.348 ] ],
    	[ [ -0.752 ], [ 1.08 ], [ 0.308 ] ],
    	[ [ 1.632 ], [ 1.448 ], [ 1.62 ] ]
    ],
    [
    	[ [ 1.888 ], [ -0.796 ], [ -0.232 ] ],
    	[ [ 1.044 ], [ -0.9 ], [ -1.32 ] ],
    	[ [ -0.76 ], [ -1.516 ], [ -0.996 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.035817870719086674, negative=3, min=1.62, max=1.62, mean=0.5853333333333333, count=9.0, positive=6, stdDev=1.0117303110128817, zeros=0},
    {meanExponent=-0.033811987344159786, negative=7, min=-0.996, max=-0.996, mean=-0.3986666666666667, count=9.0, positive=2, stdDev=1.0714360042070226, zeros=0}
    Output: [ 4.3001351111111115 ]
    Outputs Statistics: {meanExponent=0.6334821014140406, negative=0, min=4.3001351111111115, max=4.3001351111111115, mean=4.3001351111111115, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -0.344 ], [ -1.072 ], [ 1.348 ] ],
    	[ [ -0.752 ], [ 1.08 ], [ 0.308 ] ],
    	[ [ 1.632 ], [ 1.448 ], [ 1.62 ] ]
    ]
    Value Statistics: {meanExponent=-0.035817870719086674, negative=3, min=1
```
...[skipping 2360 bytes](etc/124.txt)...
```
    9154 ], [ 0.061344444439725976 ], [ -0.43998888888729937 ], [ -0.6586555555632145 ], [ -0.35110000000670993 ], [ -0.36176666666776214 ], [ -0.5813222222261771 ] ]
    Measured Statistics: {meanExponent=-0.42885683258761637, negative=6, min=-0.5813222222261771, max=-0.5813222222261771, mean=-0.2186555555589504, count=9.0, positive=3, stdDev=0.4056319021674474, zeros=0}
    Feedback Error: [ [ 1.1111108529748837E-5 ], [ 1.1111112158235237E-5 ], [ 1.1111104640137981E-5 ], [ 1.1111106392638903E-5 ], [ 1.1111112700579184E-5 ], [ 1.1111103452088322E-5 ], [ 1.1111104401162475E-5 ], [ 1.1111110015671333E-5 ], [ 1.1111107156236422E-5 ] ]
    Error Statistics: {meanExponent=-4.9542426421315335, negative=0, min=1.1111107156236422E-5, max=1.1111107156236422E-5, mean=1.1111107716277632E-5, count=9.0, positive=9, stdDev=3.183231456205249E-12, zeros=0}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.1111e-05 +- 4.7395e-12 [1.1111e-05 - 1.1111e-05] (18#)
    relativeTol: 2.0883e-05 +- 2.4757e-05 [8.4345e-06 - 9.0588e-05] (18#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.1111e-05 +- 4.7395e-12 [1.1111e-05 - 1.1111e-05] (18#), relativeTol=2.0883e-05 +- 2.4757e-05 [8.4345e-06 - 9.0588e-05] (18#)}
```



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
      "class": "com.simiacryptus.mindseye.layers.cudnn.MeanSqLossLayer",
      "id": "be77fda5-2630-4024-9323-7f5b2349f251",
      "isFrozen": false,
      "name": "MeanSqLossLayer/be77fda5-2630-4024-9323-7f5b2349f251",
      "inputs": [
        "12ff25ba-501f-4a71-9e92-c783d29d7ce8",
        "672632ad-0a9b-4146-acc4-9d7f52f37613"
      ],
      "nodes": {
        "2edeeaaf-fc2a-473d-8705-b5535008343c": "5186ce69-e5ff-4824-b07e-bf55c38369ca",
        "38a936d7-ed52-439a-bda4-68b9f5584206": "f5706cf7-d07e-4be1-ba74-99e65350cad9",
        "f555985e-9858-474c-b3a5-10c65e474246": "f4523f32-f720-4abc-8a84-2cde46efdbfb",
        "be4d058c-8955-469e-a818-672cbc325517": "4320f1b7-3f3b-415e-be21-a8de3ff05526"
      },
      "layers": {
        "5186ce69-e5ff-4824-b07e-bf55c38369ca": {
          "class": "com.simiacryptus.mindseye.layers.cudnn.BinarySumLayer",
          "id": "5186ce69-e5ff-4824-b07e-bf55c38369ca",
          "isFrozen": false,
          "name": "BinarySumLayer/5186ce69-e5ff-4824-b07e-bf55c38369ca",
          "rightFactor": -1.0,
          "leftFactor": 1.0
        },
        "f5706cf7-d07e-4be1
```
...[skipping 352 bytes](etc/125.txt)...
```
    cerLayer",
          "id": "f4523f32-f720-4abc-8a84-2cde46efdbfb",
          "isFrozen": false,
          "name": "BandReducerLayer/f4523f32-f720-4abc-8a84-2cde46efdbfb",
          "mode": 2
        },
        "4320f1b7-3f3b-415e-be21-a8de3ff05526": {
          "class": "com.simiacryptus.mindseye.layers.java.AvgReducerLayer",
          "id": "4320f1b7-3f3b-415e-be21-a8de3ff05526",
          "isFrozen": false,
          "name": "AvgReducerLayer/4320f1b7-3f3b-415e-be21-a8de3ff05526"
        }
      },
      "links": {
        "2edeeaaf-fc2a-473d-8705-b5535008343c": [
          "12ff25ba-501f-4a71-9e92-c783d29d7ce8",
          "672632ad-0a9b-4146-acc4-9d7f52f37613"
        ],
        "38a936d7-ed52-439a-bda4-68b9f5584206": [
          "2edeeaaf-fc2a-473d-8705-b5535008343c",
          "2edeeaaf-fc2a-473d-8705-b5535008343c"
        ],
        "f555985e-9858-474c-b3a5-10c65e474246": [
          "38a936d7-ed52-439a-bda4-68b9f5584206"
        ],
        "be4d058c-8955-469e-a818-672cbc325517": [
          "f555985e-9858-474c-b3a5-10c65e474246"
        ]
      },
      "labels": {},
      "head": "be4d058c-8955-469e-a818-672cbc325517"
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
    [[
    	[ [ -0.024 ], [ -0.632 ], [ 1.892 ] ],
    	[ [ -1.732 ], [ -1.08 ], [ 0.664 ] ],
    	[ [ -1.6 ], [ -0.676 ], [ -1.336 ] ]
    ],
    [
    	[ [ -1.228 ], [ -1.452 ], [ -1.78 ] ],
    	[ [ 0.0 ], [ 0.184 ], [ 0.052 ] ],
    	[ [ 1.92 ], [ 1.336 ], [ -1.836 ] ]
    ]]
    --------------------
    Output: 
    [ 4.140689777777777 ]
    --------------------
    Derivative: 
    [
    	[ [ 0.26755555555555555 ], [ 0.1822222222222222 ], [ 0.8159999999999998 ] ],
    	[ [ -0.3848888888888889 ], [ -0.28088888888888885 ], [ 0.13599999999999998 ] ],
    	[ [ -0.7822222222222222 ], [ -0.44711111111111107 ], [ 0.1111111111111111 ] ]
    ],
    [
    	[ [ -0.26755555555555555 ], [ -0.1822222222222222 ], [ -0.8159999999999998 ] ],
    	[ [ 0.3848888888888889 ], [ 0.28088888888888885 ], [ -0.13599999999999998 ] ],
    	[ [ 0.7822222222222222 ], [ 0.44711111111111107 ], [ -0.1111111111111111 ] ]
    ]
```



### Input Learning
In this test, we use a network to learn this target input, given it's pre-evaluated output:

Code from [LearningTester.java:127](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L127) executed in 0.01 seconds: 
```java
    return Arrays.stream(input_target).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ 0.296, -0.424, -1.284 ], [ -0.516, 0.652, -1.584 ], [ -0.368, 0.788, 0.488 ], [ -1.18, -0.084, 1.372 ], [ 0.176, 0.564, 1.58 ], [ 0.46, 0.112, 1.836 ], [ -1.52, 1.596, -0.368 ], [ 0.376, 0.336, -0.148 ], ... ],
    	[ [ 0.532, 0.204, -0.932 ], [ 0.66, -0.684, -1.32 ], [ -0.524, 0.656, 1.26 ], [ 1.764, 0.048, -0.812 ], [ -1.364, -1.664, -1.164 ], [ 0.424, 1.312, -1.528 ], [ 0.664, 0.944, 1.452 ], [ 0.236, 1.164, -0.836 ], ... ],
    	[ [ 1.62, 1.996, 1.596 ], [ -0.352, 1.312, -0.428 ], [ -0.608, -0.256, -0.616 ], [ -0.788, -1.432, -1.824 ], [ 0.764, 0.952, -1.068 ], [ 0.912, 0.308, 1.928 ], [ -0.196, -0.3, 1.048 ], [ 0.868, -1.512, 1.976 ], ... ],
    	[ [ 0.468, -0.812, -0.444 ], [ 1.976, 0.596, -0.512 ], [ 0.82, -1.412, -0.092 ], [ 0.116, -0.292, -1.744 ], [ -1.552, 0.0, 1.804 ], [ 0.484, 0.48, -0.488 ], [ -0.732, -1.556, -0.796 ], [ 0.912, -0.34, 0.192 ], ... ],
    	[ [ -0.832, 1.872, -1.432 ], [ -0.644, -0.984, 1.244 ], [ -1.592, 0.672, 0.784 ], [ -1.12, 0.036, -0.492 ], [ -1.156, -1.688, -1.4 ], [ 1.316, -1.736, 
```
...[skipping 1448 bytes](etc/126.txt)...
```
    0.064 ], [ 0.344, 1.988, 1.428 ], [ -0.056, 0.636, -2.0 ], [ 1.476, 1.712, 0.164 ], [ -0.112, 0.596, -1.984 ], [ 0.656, -0.252, -0.932 ], ... ],
    	[ [ 0.576, -0.1, 1.728 ], [ 1.164, -0.796, 0.128 ], [ 1.68, 1.512, -0.832 ], [ -1.292, 0.76, 0.712 ], [ 1.744, 1.596, 0.984 ], [ 1.748, -0.004, 1.748 ], [ -1.132, -1.38, -0.344 ], [ -1.924, 0.872, 1.82 ], ... ],
    	[ [ 0.452, 0.132, 0.028 ], [ 1.136, -1.516, 0.972 ], [ 1.396, -1.068, -0.208 ], [ 0.012, 1.424, -1.028 ], [ -0.124, -1.64, 1.956 ], [ 1.112, 1.412, 0.592 ], [ -1.08, -1.436, -0.708 ], [ -1.096, 0.144, -1.824 ], ... ],
    	[ [ 0.212, 0.548, -1.148 ], [ -0.42, 1.444, -1.488 ], [ -1.212, -1.656, 1.2 ], [ -0.208, 1.672, 0.756 ], [ -1.604, 0.744, 1.152 ], [ 1.684, -1.224, 0.772 ], [ 1.22, -1.408, -0.196 ], [ -1.764, -0.412, -0.448 ], ... ],
    	[ [ 0.872, 0.288, -1.844 ], [ 1.352, 0.456, -0.464 ], [ -0.54, -1.8, -0.392 ], [ -0.456, 1.556, -1.524 ], [ -1.036, -0.328, -0.784 ], [ -1.72, -0.436, 1.824 ], [ -0.684, -0.964, -0.716 ], [ 0.648, 1.288, -0.088 ], ... ],
    	...
    ]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 0.02 seconds: 
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
    F(0.0) = LineSearchPoint{point=PointSample{avg=7.0797637975421335}, derivative=0.0}
    Iteration 1 failed, aborting. Error: 7.0797637975421335 Total: 239577166802500.8400; Orientation: 0.0009; Line Search: 0.0063
    
```

Returns: 

```
    7.0797637975421335
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:144](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L144) executed in 0.01 seconds: 
```java
    return Arrays.stream(input_gd).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ 1.24, -1.892, 1.308 ], [ -1.256, -1.672, 1.188 ], [ -0.464, 1.044, -0.304 ], [ 0.184, -1.256, 1.308 ], [ -0.364, 1.7, -1.44 ], [ -1.92, -1.856, -0.656 ], [ 1.312, 0.604, 1.16 ], [ -0.652, -0.652, -0.064 ], ... ],
    	[ [ -1.5, -0.988, 1.704 ], [ -0.992, -1.068, 0.924 ], [ -1.164, -0.748, -1.636 ], [ -0.476, -1.12, -1.172 ], [ 0.42, -0.94, 1.936 ], [ 1.148, -0.616, -0.412 ], [ 1.096, -0.512, 1.852 ], [ 0.156, 0.756, 1.48 ], ... ],
    	[ [ -1.664, -0.056, 0.468 ], [ 0.824, -0.112, 0.66 ], [ -1.396, 0.144, 1.38 ], [ 1.152, -1.5, 0.248 ], [ 0.108, -1.688, -0.424 ], [ -0.78, 1.772, -0.996 ], [ -1.792, 1.988, -0.448 ], [ 0.976, -1.4, 0.824 ], ... ],
    	[ [ 1.48, -0.104, 1.032 ], [ -1.084, -0.104, -0.04 ], [ 1.756, -1.636, -1.268 ], [ 0.272, 1.968, -0.38 ], [ -1.632, 0.996, -0.692 ], [ -1.844, 0.432, -0.072 ], [ -1.248, -1.812, -0.892 ], [ 0.508, -0.896, -1.244 ], ... ],
    	[ [ 1.968, 0.9, 0.432 ], [ 1.128, 0.5, -1.392 ], [ -1.876, 1.404, 0.896 ], [ -1.16, -0.9, -1.756 ], [ 0.224, 1.484, -1.1 ], [ -1.856, -0.788, 0.228
```
...[skipping 1439 bytes](etc/127.txt)...
```
    92 ], [ 0.096, -1.968, -0.244 ], [ 1.9, -0.052, -1.572 ], [ 1.624, -0.432, 0.208 ], [ 1.256, -0.128, 0.828 ], [ 0.656, 0.044, -0.86 ], ... ],
    	[ [ -1.216, 0.348, -1.648 ], [ 1.092, 1.624, 1.708 ], [ 0.688, 1.952, 0.804 ], [ -0.412, 0.536, 1.536 ], [ 1.688, 1.948, 1.528 ], [ -0.396, -0.168, -1.924 ], [ -0.172, 0.992, -1.572 ], [ -1.208, -0.492, 1.48 ], ... ],
    	[ [ -0.836, -1.692, 1.312 ], [ 1.504, 0.072, -1.932 ], [ 0.448, 1.484, -1.084 ], [ -1.416, -0.596, 1.28 ], [ -1.092, -0.092, -0.36 ], [ -0.672, 1.828, 1.056 ], [ -0.468, 1.208, -0.932 ], [ -0.46, -0.976, 0.836 ], ... ],
    	[ [ -0.356, 0.92, -1.684 ], [ -0.924, 1.624, -0.372 ], [ -0.96, 0.596, -0.452 ], [ 1.504, -0.972, -0.612 ], [ -0.924, 1.716, 0.576 ], [ -1.856, -1.2, 0.756 ], [ -0.388, 1.216, -1.456 ], [ -1.112, 1.208, 1.744 ], ... ],
    	[ [ 1.736, 1.74, 0.648 ], [ 0.596, -1.496, -1.716 ], [ 1.688, -1.728, 1.18 ], [ 1.416, 0.108, -1.012 ], [ 0.756, 1.036, -0.36 ], [ -1.356, -1.136, -1.868 ], [ 0.28, -0.964, -0.56 ], [ 0.244, -1.044, 0.348 ], ... ],
    	...
    ]
```



Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 0.03 seconds: 
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
    th(0)=7.0797637975421335;dx=0.0 (ERROR: Starting derivative negative)
    Iteration 1 failed, aborting. Error: 7.0797637975421335 Total: 239577210404478.8000; Orientation: 0.0015; Line Search: 0.0138
    
```

Returns: 

```
    7.0797637975421335
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:154](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L154) executed in 0.01 seconds: 
```java
    return Arrays.stream(input_lbgfs).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ 1.24, -1.892, 1.308 ], [ -1.256, -1.672, 1.188 ], [ -0.464, 1.044, -0.304 ], [ 0.184, -1.256, 1.308 ], [ -0.364, 1.7, -1.44 ], [ -1.92, -1.856, -0.656 ], [ 1.312, 0.604, 1.16 ], [ -0.652, -0.652, -0.064 ], ... ],
    	[ [ -1.5, -0.988, 1.704 ], [ -0.992, -1.068, 0.924 ], [ -1.164, -0.748, -1.636 ], [ -0.476, -1.12, -1.172 ], [ 0.42, -0.94, 1.936 ], [ 1.148, -0.616, -0.412 ], [ 1.096, -0.512, 1.852 ], [ 0.156, 0.756, 1.48 ], ... ],
    	[ [ -1.664, -0.056, 0.468 ], [ 0.824, -0.112, 0.66 ], [ -1.396, 0.144, 1.38 ], [ 1.152, -1.5, 0.248 ], [ 0.108, -1.688, -0.424 ], [ -0.78, 1.772, -0.996 ], [ -1.792, 1.988, -0.448 ], [ 0.976, -1.4, 0.824 ], ... ],
    	[ [ 1.48, -0.104, 1.032 ], [ -1.084, -0.104, -0.04 ], [ 1.756, -1.636, -1.268 ], [ 0.272, 1.968, -0.38 ], [ -1.632, 0.996, -0.692 ], [ -1.844, 0.432, -0.072 ], [ -1.248, -1.812, -0.892 ], [ 0.508, -0.896, -1.244 ], ... ],
    	[ [ 1.968, 0.9, 0.432 ], [ 1.128, 0.5, -1.392 ], [ -1.876, 1.404, 0.896 ], [ -1.16, -0.9, -1.756 ], [ 0.224, 1.484, -1.1 ], [ -1.856, -0.788, 0.228
```
...[skipping 1439 bytes](etc/128.txt)...
```
    92 ], [ 0.096, -1.968, -0.244 ], [ 1.9, -0.052, -1.572 ], [ 1.624, -0.432, 0.208 ], [ 1.256, -0.128, 0.828 ], [ 0.656, 0.044, -0.86 ], ... ],
    	[ [ -1.216, 0.348, -1.648 ], [ 1.092, 1.624, 1.708 ], [ 0.688, 1.952, 0.804 ], [ -0.412, 0.536, 1.536 ], [ 1.688, 1.948, 1.528 ], [ -0.396, -0.168, -1.924 ], [ -0.172, 0.992, -1.572 ], [ -1.208, -0.492, 1.48 ], ... ],
    	[ [ -0.836, -1.692, 1.312 ], [ 1.504, 0.072, -1.932 ], [ 0.448, 1.484, -1.084 ], [ -1.416, -0.596, 1.28 ], [ -1.092, -0.092, -0.36 ], [ -0.672, 1.828, 1.056 ], [ -0.468, 1.208, -0.932 ], [ -0.46, -0.976, 0.836 ], ... ],
    	[ [ -0.356, 0.92, -1.684 ], [ -0.924, 1.624, -0.372 ], [ -0.96, 0.596, -0.452 ], [ 1.504, -0.972, -0.612 ], [ -0.924, 1.716, 0.576 ], [ -1.856, -1.2, 0.756 ], [ -0.388, 1.216, -1.456 ], [ -1.112, 1.208, 1.744 ], ... ],
    	[ [ 1.736, 1.74, 0.648 ], [ 0.596, -1.496, -1.716 ], [ 1.688, -1.728, 1.18 ], [ 1.416, 0.108, -1.012 ], [ 0.756, 1.036, -0.36 ], [ -1.356, -1.136, -1.868 ], [ 0.28, -0.964, -0.56 ], [ 0.244, -1.044, 0.348 ], ... ],
    	...
    ]
```



Code from [LearningTester.java:95](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.00 seconds: 
```java
    return TestUtil.compare(runs);
```

Code from [LearningTester.java:98](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

### Performance
Adding performance wrappers

Code from [TestUtil.java:287](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/TestUtil.java#L287) executed in 0.00 seconds: 
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

Code from [PerformanceTester.java:66](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/PerformanceTester.java#L66) executed in 0.76 seconds: 
```java
    test(component, inputPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[100, 100, 3]
    	[100, 100, 3]
    Performance:
    	Evaluation performance: 0.082334s +- 0.046426s [0.056669s - 0.174999s]
    	Learning performance: 0.000378s +- 0.000081s [0.000280s - 0.000499s]
    
```

Per-layer Performance Metrics:

Code from [TestUtil.java:252](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/TestUtil.java#L252) executed in 0.00 seconds: 
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
    	Optional[BandReducerLayer/f4523f32-f720-4abc-8a84-2cde46efdbfb -> 0.025381s +- 0.015613s (11.0)
    	ProductLayer/f5706cf7-d07e-4be1-ba74-99e65350cad9 -> 0.001342s +- 0.000329s (11.0)
    	BinarySumLayer/5186ce69-e5ff-4824-b07e-bf55c38369ca -> 0.037356s +- 0.019608s (11.0)
    	AvgReducerLayer/4320f1b7-3f3b-415e-be21-a8de3ff05526 -> 0.003687s +- 0.009741s (11.0)]
    Backward Performance: 
    	Optional[BandReducerLayer/f4523f32-f720-4abc-8a84-2cde46efdbfb -> 0.000021s +- 0.000044s (6.0)
    	ProductLayer/f5706cf7-d07e-4be1-ba74-99e65350cad9 -> 0.000917s +- 0.000000s (1.0)
    	BinarySumLayer/5186ce69-e5ff-4824-b07e-bf55c38369ca -> 0.000541s +- 0.000000s (1.0)
    	AvgReducerLayer/4320f1b7-3f3b-415e-be21-a8de3ff05526 -> 0.000007s +- 0.000011s (6.0)]
    
```

Removing performance wrappers

Code from [TestUtil.java:270](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/TestUtil.java#L270) executed in 0.00 seconds: 
```java
    network.visitNodes(node -> {
      if (node.getLayer() instanceof MonitoringWrapperLayer) {
        node.setLayer(node.<MonitoringWrapperLayer>getLayer().getInner());
      }
    });
```

