# ConvolutionLayer
## ConvolutionLayerTest
### Json Serialization
Code from [StandardLayerTests.java:69](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L69) executed in 0.04 seconds: 
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
      "id": "a765d2ce-adba-415e-865b-afd3c4780089",
      "isFrozen": false,
      "name": "ConvolutionLayer/a765d2ce-adba-415e-865b-afd3c4780089",
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
            0.94
          ]
        ]
      ],
      "skip": [
        [
          0.0
        ]
      ],
      "simple": true
    }
```



### Example Input/Output Pair
Code from [StandardLayerTests.java:153](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L153) executed in 2.88 seconds: 
```java
    SimpleEval eval = SimpleEval.run(layer, inputPrototype);
    return String.format("--------------------\nInput: \n[%s]\n--------------------\nOutput: \n%s\n--------------------\nDerivative: \n%s",
      Arrays.stream(inputPrototype).map(t -> t.prettyPrint()).reduce((a, b) -> a + ",\n" + b).get(),
      eval.getOutput().prettyPrint(),
      Arrays.stream(eval.getDerivative()).map(t -> t.prettyPrint()).reduce((a, b) -> a + ",\n" + b).get());
```
Logging: 
```
    Found 2 devices
    Device 0 - GeForce GTX 1080 Ti
    Device 1 - GeForce GTX 1060 6GB
    Found 2 devices; using devices [0, 1]
    
```

Returns: 

```
    --------------------
    Input: 
    [[
    	[ [ 1.268, -0.392 ], [ 0.2, -0.52 ], [ -1.604, -1.468 ] ],
    	[ [ -1.556, -0.096 ], [ 0.152, 1.776 ], [ 1.852, -0.196 ] ],
    	[ [ 0.052, 1.024 ], [ -1.392, -0.992 ], [ -1.408, 0.808 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, -0.36848 ], [ 0.0, -0.4888 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, -0.09024 ], [ 0.0, 1.66944 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 0.0, 0.94 ], [ 0.0, 0.94 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 0.94 ], [ 0.0, 0.94 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ]
    ]
```



### Batch Execution
Code from [StandardLayerTests.java:102](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L102) executed in 0.10 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (360#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (80#)}
```



### Differential Validation
Code from [StandardLayerTests.java:110](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L110) executed in 0.55 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -1.668, 0.516 ], [ 0.232, 1.752 ], [ -1.672, -0.984 ] ],
    	[ [ 1.296, 0.172 ], [ -0.424, 0.6 ], [ 0.356, 0.416 ] ],
    	[ [ -1.52, 0.148 ], [ -1.964, 0.084 ], [ -1.108, -0.924 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.2075538391850328, negative=8, min=-0.924, max=-0.924, mean=-0.26066666666666666, count=18.0, positive=10, stdDev=1.0418856836418176, zeros=0}
    Output: [
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 0.48503999999999997 ], [ 0.0, 1.64688 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 0.16168 ], [ 0.0, 0.564 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.2844062712126849, negative=0, min=0.564, max=0.564, mean=0.15875555555555554, count=18.0, positive=4, stdDev=0.39707640482548917, zeros=14}
    Feedback for input 0
    Inputs Values: [
    	[ [ -1.668, 0.516 ], [ 0.232, 1.752 ], [ -1.672, -0.984 ] ],
    	[ [ 1.296, 0.172 ], [ -0.424, 0.6 ], [ 0.356, 0.416 ] ],
    	[ [ -1.52, 0.148 ], [ -1.964, 0.084 ], [ -1.108, -0.924 ] ]
    ]
    Value Statistics: {meanExponent=-0.2075538391850328, negative=8, min=-0
```
...[skipping 3248 bytes](etc/1.txt)...
```
    red Statistics: {meanExponent=-0.237000526733521, negative=78, min=0.5999999999994898, max=0.5999999999994898, mean=-0.04074074074073947, count=648.0, positive=118, stdDev=0.5643867375964396, zeros=452}
    Gradient Error: [ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], ... ]
    Error Statistics: {meanExponent=-13.50603787891276, negative=27, min=-5.101474798152594E-13, max=-5.101474798152594E-13, mean=1.2759854901073848E-15, count=648.0, positive=37, stdDev=1.0973015718700925E-13, zeros=584}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.5875e-14 +- 9.0226e-14 [0.0000e+00 - 1.1303e-12] (972#)
    relativeTol: 8.0893e-14 +- 2.3386e-13 [0.0000e+00 - 1.3329e-12] (200#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.5875e-14 +- 9.0226e-14 [0.0000e+00 - 1.1303e-12] (972#), relativeTol=8.0893e-14 +- 2.3386e-13 [0.0000e+00 - 1.3329e-12] (200#)}
```



### Performance
Code from [StandardLayerTests.java:120](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L120) executed in 0.04 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    Evaluation performance: 0.002368s +- 0.000864s [0.001687s - 0.004052s]
    Learning performance: 0.001693s +- 0.000688s [0.001242s - 0.003059s]
    
```

