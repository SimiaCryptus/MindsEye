# ImgConcatLayer
## Float
### Json Serialization
Code from [StandardLayerTests.java:69](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L69) executed in 0.00 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.ImgConcatLayer",
      "id": "fcd2c198-a75f-49a4-8ab0-fc7299bb3b23",
      "isFrozen": false,
      "name": "ImgConcatLayer/fcd2c198-a75f-49a4-8ab0-fc7299bb3b23",
      "maxBands": -1
    }
```



### Example Input/Output Pair
Code from [StandardLayerTests.java:153](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L153) executed in 0.00 seconds: 
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
    	[ [ -1.828 ], [ -0.42 ] ],
    	[ [ -1.088 ], [ -1.456 ] ]
    ],
    [
    	[ [ 1.452 ], [ -1.704 ] ],
    	[ [ -1.62 ], [ 1.3 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -1.828, 1.452 ], [ -0.42, -1.704 ] ],
    	[ [ -1.088, -1.62 ], [ -1.456, 1.3 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 1.0 ], [ 1.0 ] ],
    	[ [ 1.0 ], [ 1.0 ] ]
    ],
    [
    	[ [ 1.0 ], [ 1.0 ] ],
    	[ [ 1.0 ], [ 1.0 ] ]
    ]
```



[GPU Log](etc/cuda.log)

### Batch Execution
Code from [StandardLayerTests.java:102](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L102) executed in 0.01 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (160#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (160#)}
```



### Differential Validation
Code from [StandardLayerTests.java:110](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L110) executed in 0.02 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -0.016 ], [ -0.076 ] ],
    	[ [ -1.828 ], [ -0.74 ] ]
    ],
    [
    	[ [ -1.868 ], [ 1.452 ] ],
    	[ [ -0.048 ], [ 1.508 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.6959646284836237, negative=4, min=-0.74, max=-0.74, mean=-0.665, count=4.0, positive=0, stdDev=0.7290946440620724, zeros=0},
    {meanExponent=-0.176753483208127, negative=2, min=1.508, max=1.508, mean=0.26099999999999995, count=4.0, positive=2, stdDev=1.3785510509226708, zeros=0}
    Output: [
    	[ [ -0.016, -1.868 ], [ -0.076, 1.452 ] ],
    	[ [ -1.828, -0.048 ], [ -0.74, 1.508 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.43635905584587537, negative=6, min=1.508, max=1.508, mean=-0.20200000000000007, count=8.0, positive=2, stdDev=1.1959765883996225, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -0.016 ], [ -0.076 ] ],
    	[ [ -1.828 ], [ -0.74 ] ]
    ]
    Value Statistics: {meanExponent=-0.6959646284836237, negative=4, min=-0.74, max=-0.74, mean=-0.665, count=4.0, positive=0, stdDev=0.7290946440620724, zeros=0}
    Implemented Feedback: [ [ 1.0, 0.0, 0.0, 0.0, 
```
...[skipping 1924 bytes](etc/36.txt)...
```
    000000286, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9999999999998899, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9999999999998899 ] ]
    Measured Statistics: {meanExponent=-3.276302567614995E-14, negative=0, min=0.9999999999998899, max=0.9999999999998899, mean=0.12499999999999056, count=32.0, positive=4, stdDev=0.33071891388304886, zeros=28}
    Feedback Error: [ [ 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 2.864375403532904E-14, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.1013412404281553E-13 ] ]
    Error Statistics: {meanExponent=-13.104301089584563, negative=3, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=-9.429956815409923E-15, count=32.0, positive=1, stdDev=3.2769779210413227E-14, zeros=28}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 9.5930e-15 +- 2.9695e-14 [0.0000e+00 - 1.1013e-13] (64#)
    relativeTol: 3.8372e-14 +- 2.1800e-14 [2.9976e-15 - 5.5067e-14] (8#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=9.5930e-15 +- 2.9695e-14 [0.0000e+00 - 1.1013e-13] (64#), relativeTol=3.8372e-14 +- 2.1800e-14 [2.9976e-15 - 5.5067e-14] (8#)}
```



### Performance
Code from [StandardLayerTests.java:120](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L120) executed in 0.01 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    Evaluation performance: 0.000288s +- 0.000044s [0.000252s - 0.000374s]
    Learning performance: 0.000235s +- 0.000024s [0.000219s - 0.000282s]
    
```

