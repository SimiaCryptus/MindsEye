# RescaledSubnetLayer
## Normal
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
      "class": "com.simiacryptus.mindseye.layers.java.RescaledSubnetLayer",
      "id": "4f637d1c-c4b9-41d5-9a81-8dab82d17902",
      "isFrozen": false,
      "name": "RescaledSubnetLayer/4f637d1c-c4b9-41d5-9a81-8dab82d17902",
      "scale": 2,
      "subnetwork": {
        "class": "com.simiacryptus.mindseye.layers.cudnn.ConvolutionLayer",
        "id": "622ffc71-e951-41d2-bfde-27206bba5aaf",
        "isFrozen": false,
        "name": "ConvolutionLayer/622ffc71-e951-41d2-bfde-27206bba5aaf",
        "filter": [
          [
            [
              0.0,
              0.0,
              0.0
            ],
            [
              0.0,
              0.0,
              0.0
            ],
            [
              0.0,
              0.0,
              0.0
            ]
          ]
        ],
        "strideX": 1,
        "strideY": 1
      }
    }
```



### Example Input/Output Pair
Code from [StandardLayerTests.java:153](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L153) executed in 0.01 seconds: 
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
    	[ [ 1.644 ], [ 1.744 ], [ -1.576 ], [ -0.832 ], [ -1.884 ], [ 1.232 ] ],
    	[ [ 1.244 ], [ 0.684 ], [ -0.804 ], [ 1.068 ], [ -0.504 ], [ -0.34 ] ],
    	[ [ 1.228 ], [ -1.068 ], [ -0.52 ], [ 0.776 ], [ 1.748 ], [ -0.692 ] ],
    	[ [ -1.044 ], [ 0.616 ], [ 0.92 ], [ -0.852 ], [ 1.688 ], [ 0.028 ] ],
    	[ [ 0.016 ], [ 1.916 ], [ -0.36 ], [ 0.696 ], [ 0.62 ], [ 1.548 ] ],
    	[ [ -1.612 ], [ -0.22 ], [ -0.216 ], [ 1.4 ], [ -1.612 ], [ 0.212 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ] ]
    ]
```



### Batch Execution
Code from [StandardLayerTests.java:102](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L102) executed in 0.06 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (720#), relativeTol=0.0000e+00 +- 0.0000e+00 [Infinity - -Infinity] (0#)}
```



### Differential Validation
Code from [StandardLayerTests.java:110](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L110) executed in 0.35 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 1.468 ], [ -1.076 ], [ 1.38 ], [ -1.5 ], [ 1.0 ], [ -0.156 ] ],
    	[ [ -0.556 ], [ -0.636 ], [ 1.596 ], [ 1.952 ], [ 0.556 ], [ 0.784 ] ],
    	[ [ -1.58 ], [ -1.224 ], [ -1.032 ], [ 1.684 ], [ -0.6 ], [ -0.928 ] ],
    	[ [ 0.032 ], [ -0.516 ], [ 1.316 ], [ -0.528 ], [ -0.964 ], [ 0.788 ] ],
    	[ [ -0.192 ], [ -1.028 ], [ 1.444 ], [ -0.532 ], [ -0.684 ], [ -1.004 ] ],
    	[ [ -0.112 ], [ 1.188 ], [ -0.016 ], [ 1.412 ], [ 0.376 ], [ -0.116 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.20175780689447398, negative=21, min=-0.116, max=-0.116, mean=0.055444444444444435, count=36.0, positive=15, stdDev=1.0251153443080452, zeros=0}
    Output: [
    	[ [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ] ]
    ]
    Outputs Statistics: {meanE
```
...[skipping 1569 bytes](etc/98.txt)...
```
    , 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], ... ]
    Measured Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=1296.0, positive=0, stdDev=0.0, zeros=1296}
    Feedback Error: [ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], ... ]
    Error Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=1296.0, positive=0, stdDev=0.0, zeros=1296}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (1296#)
    relativeTol: 0.0000e+00 +- 0.0000e+00 [Infinity - -Infinity] (0#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (1296#), relativeTol=0.0000e+00 +- 0.0000e+00 [Infinity - -Infinity] (0#)}
```



### Performance
Code from [StandardLayerTests.java:120](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L120) executed in 0.08 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    Evaluation performance: 0.004794s +- 0.000304s [0.004411s - 0.005163s]
    Learning performance: 0.004532s +- 0.000447s [0.003906s - 0.005006s]
    
```

