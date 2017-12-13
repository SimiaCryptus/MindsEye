# ConvolutionLayer
## UpsizeTest
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
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "551466a9-f987-41d8-961e-187606dfcc3a",
      "isFrozen": false,
      "name": "ConvolutionLayer/551466a9-f987-41d8-961e-187606dfcc3a",
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
        ],
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
        ],
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
        ],
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
        ],
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
        ],
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
            -0.404
          ]
        ]
      ],
      "skip": [
        [
          0.0
        ]
      ],
      "simple": false
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
    	[ [ 0.988, 0.492 ], [ 0.78, -1.444 ], [ 1.568, -1.824 ] ],
    	[ [ -0.52, 1.596 ], [ 0.572, 1.316 ], [ 0.632, 1.056 ] ],
    	[ [ -1.132, -1.0 ], [ -1.428, 0.976 ], [ 0.472, 1.104 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.0, 0.0, 0.0 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ]
    ]
```



### Batch Execution
Code from [StandardLayerTests.java:102](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L102) executed in 0.08 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (210#), relativeTol=0.0000e+00 +- 0.0000e+00 [Infinity - -Infinity] (0#)}
```



### Differential Validation
Code from [StandardLayerTests.java:110](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L110) executed in 0.30 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -1.964, 1.656 ], [ 0.872, 0.308 ], [ 1.504, -1.636 ] ],
    	[ [ 1.628, -1.824 ], [ 0.088, 1.484 ], [ -1.86, 0.26 ] ],
    	[ [ -0.12, 1.832 ], [ 1.588, 1.836 ], [ -0.356, 1.62 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.04593460246652856, negative=6, min=1.62, max=1.62, mean=0.3842222222222222, count=18.0, positive=12, stdDev=1.3635140693384686, zeros=0}
    Output: [
    	[ [ 0.0, 0.0, 0.0 ] ]
    ]
    Outputs Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=3.0, positive=0, stdDev=0.0, zeros=3}
    Feedback for input 0
    Inputs Values: [
    	[ [ -1.964, 1.656 ], [ 0.872, 0.308 ], [ 1.504, -1.636 ] ],
    	[ [ 1.628, -1.824 ], [ 0.088, 1.484 ], [ -1.86, 0.26 ] ],
    	[ [ -0.12, 1.832 ], [ 1.588, 1.836 ], [ -0.356, 1.62 ] ]
    ]
    Value Statistics: {meanExponent=-0.04593460246652856, negative=6, min=1.62, max=1.62, mean=0.3842222222222222, count=18.0, positive=12, stdDev=1.3635140693384686, zeros=0}
    Implemented Feedback: [ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0,
```
...[skipping 1098 bytes](etc/4.txt)...
```
    lemented Statistics: {meanExponent=0.25610090794989604, negative=3, min=0.0, max=0.0, mean=-0.005703703703703705, count=162.0, positive=3, stdDev=0.3495466709489131, zeros=156}
    Measured Gradient: [ [ -1.964, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], ... ]
    Measured Statistics: {meanExponent=0.25610090794989604, negative=3, min=0.0, max=0.0, mean=-0.005703703703703705, count=162.0, positive=3, stdDev=0.3495466709489131, zeros=156}
    Gradient Error: [ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], ... ]
    Error Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=162.0, positive=0, stdDev=0.0, zeros=162}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (216#)
    relativeTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (6#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (216#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (6#)}
```



### Performance
Code from [StandardLayerTests.java:120](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L120) executed in 0.04 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    Evaluation performance: 0.002355s +- 0.000302s [0.002129s - 0.002932s]
    Learning performance: 0.001376s +- 0.000524s [0.001082s - 0.002422s]
    
```

