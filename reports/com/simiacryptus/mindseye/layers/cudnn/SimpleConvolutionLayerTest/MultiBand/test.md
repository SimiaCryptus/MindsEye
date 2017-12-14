# SimpleConvolutionLayer
## MultiBand
### Json Serialization
Code from [StandardLayerTests.java:68](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L68) executed in 0.00 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.SimpleConvolutionLayer",
      "id": "79ef6766-f72f-40dd-8e79-073013dde7ee",
      "isFrozen": false,
      "name": "SimpleConvolutionLayer/79ef6766-f72f-40dd-8e79-073013dde7ee",
      "filter": [
        [
          [
            1.688
          ]
        ],
        [
          [
            1.52
          ]
        ],
        [
          [
            1.668
          ]
        ],
        [
          [
            1.968
          ]
        ],
        [
          [
            -1.868
          ]
        ],
        [
          [
            1.484
          ]
        ],
        [
          [
            1.656
          ]
        ],
        [
          [
            0.016
          ]
        ],
        [
          [
            -1.088
          ]
        ]
      ],
      "strideX": 1,
      "strideY": 1,
      "simple": false
    }
```



### Example Input/Output Pair
Code from [StandardLayerTests.java:152](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L152) executed in 0.00 seconds: 
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
    	[ [ 1.596, 0.98, 1.688 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 6.999231999999999, 3.81528, 0.8221120000000001 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 5.311999999999999, -0.3320000000000001, 2.064 ] ]
    ]
```



[GPU Log](etc/cuda.log)

### Reference Implementation
Code from [StandardLayerTests.java:92](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L92) executed in 0.01 seconds: 
```java
    System.out.println(new GsonBuilder().setPrettyPrinting().create().toJson(referenceLayer.getJson()));
    getEquivalencyTester().test(referenceLayer, layer, inputPrototype);
```
Logging: 
```
    {
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "da11710a-088d-483b-9c05-e73c7ed0d959",
      "isFrozen": false,
      "name": "ConvolutionLayer/da11710a-088d-483b-9c05-e73c7ed0d959",
      "filter": [
        [
          [
            1.688
          ]
        ],
        [
          [
            1.968
          ]
        ],
        [
          [
            1.656
          ]
        ],
        [
          [
            1.52
          ]
        ],
        [
          [
            -1.868
          ]
        ],
        [
          [
            0.016
          ]
        ],
        [
          [
            1.668
          ]
        ],
        [
          [
            1.484
          ]
        ],
        [
          [
            -1.088
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
    Inputs: [
    	[ [ 0.028, -0.688, -0.08 ] ]
    ]
    Error: [
    	[ [ 0.0, 0.0, 0.0 ] ]
    ]
    Accuracy:
    absoluteTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (3#)
    relativeTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (3#)
    
```

### Batch Execution
Code from [StandardLayerTests.java:101](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L101) executed in 0.01 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (60#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (60#)}
```



### Differential Validation
Code from [StandardLayerTests.java:109](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L109) executed in 0.02 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 1.952, 1.464, 1.496 ] ]
    ]
    Inputs Statistics: {meanExponent=0.21031749452716286, negative=0, min=1.496, max=1.496, mean=1.6373333333333333, count=3.0, positive=3, stdDev=0.22288611940231287, zeros=0}
    Output: [
    	[ [ 8.015584, 3.326848, 1.6282879999999995 ] ]
    ]
    Outputs Statistics: {meanExponent=0.5458997836187404, negative=0, min=1.6282879999999995, max=1.6282879999999995, mean=4.323573333333333, count=3.0, positive=3, stdDev=2.701170559897484, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 1.952, 1.464, 1.496 ] ]
    ]
    Value Statistics: {meanExponent=0.21031749452716286, negative=0, min=1.496, max=1.496, mean=1.6373333333333333, count=3.0, positive=3, stdDev=0.22288611940231287, zeros=0}
    Implemented Feedback: [ [ 1.688, 1.968, 1.656 ], [ 1.52, -1.868, 0.016 ], [ 1.668, 1.484, -1.088 ] ]
    Implemented Statistics: {meanExponent=-0.019104760673835392, negative=2, min=-1.088, max=-1.088, mean=0.7826666666666666, count=9.0, positive=7, stdDev=1.3299918128402979, zeros=0}
    Measured Feedback: [ [
```
...[skipping 1624 bytes](etc/90.txt)...
```
     [ 0.0, 0.0, 1.4640000000021303 ], [ 0.0, 0.0, 1.49600000000083 ] ]
    Measured Statistics: {meanExponent=0.21031749452672555, negative=0, min=1.49600000000083, max=1.49600000000083, mean=0.545777777777271, count=27.0, positive=9, stdDev=0.7824999506955425, zeros=18}
    Gradient Error: [ [ 2.8403945862010005E-12, 0.0, 0.0 ], [ 2.1302959396507504E-12, 0.0, 0.0 ], [ -1.4713119611542425E-11, 0.0, 0.0 ], [ 0.0, -1.6004975122996257E-12, 0.0 ], [ 0.0, -2.3105961588498758E-12, 0.0 ], [ 0.0, -1.390443316040546E-12, 0.0 ], [ 0.0, 0.0, -1.6004975122996257E-12 ], [ 0.0, 0.0, 2.1302959396507504E-12 ], [ 0.0, 0.0, 8.30002733209767E-13 ] ]
    Error Statistics: {meanExponent=-11.654174423752586, negative=5, min=8.30002733209767E-13, max=8.30002733209767E-13, mean=-5.068209226785122E-13, count=27.0, positive=4, stdDev=2.979988235983592E-12, zeros=18}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.6502e-12 +- 3.2985e-12 [0.0000e+00 - 1.4713e-11] (36#)
    relativeTol: 1.8243e-12 +- 3.2984e-12 [1.6339e-13 - 1.4378e-11] (18#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.6502e-12 +- 3.2985e-12 [0.0000e+00 - 1.4713e-11] (36#), relativeTol=1.8243e-12 +- 3.2984e-12 [1.6339e-13 - 1.4378e-11] (18#)}
```



### Performance
Code from [StandardLayerTests.java:119](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L119) executed in 0.95 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[100, 100, 3]
    Performance:
    	Evaluation performance: 0.021326s +- 0.001869s [0.019038s - 0.023592s]
    	Learning performance: 0.143212s +- 0.079671s [0.078825s - 0.300313s]
    
```

