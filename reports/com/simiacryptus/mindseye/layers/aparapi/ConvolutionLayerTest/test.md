# ConvolutionLayer
## ConvolutionLayerTest
### Json Serialization
Code from [StandardLayerTests.java:68](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L68) executed in 0.00 seconds: 
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
      "id": "6aa547a1-7d34-4f9a-9764-7fc043f78fd3",
      "isFrozen": false,
      "name": "ConvolutionLayer/6aa547a1-7d34-4f9a-9764-7fc043f78fd3",
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
            1.92
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
Code from [StandardLayerTests.java:152](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L152) executed in 0.44 seconds: 
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
    	[ [ -1.692, 1.328 ], [ 0.808, -1.412 ], [ -1.928, -1.272 ] ],
    	[ [ -1.328, 0.756 ], [ 1.676, -0.888 ], [ -1.832, -1.572 ] ],
    	[ [ 0.708, 1.524 ], [ 0.232, -1.348 ], [ -1.952, -0.328 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 2.54976 ], [ 0.0, -2.7110399999999997 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 1.45152 ], [ 0.0, -1.70496 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 0.0, 1.92 ], [ 0.0, 1.92 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 1.92 ], [ 0.0, 1.92 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ]
    ]
```



### Batch Execution
Code from [StandardLayerTests.java:101](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L101) executed in 0.12 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (360#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (80#)}
```



### Differential Validation
Code from [StandardLayerTests.java:109](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L109) executed in 0.49 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 0.132, -1.56 ], [ 1.816, 0.528 ], [ 1.604, -0.348 ] ],
    	[ [ 1.156, 0.268 ], [ 0.736, 0.672 ], [ -0.836, 0.448 ] ],
    	[ [ -1.916, -0.052 ], [ -0.656, 1.032 ], [ -1.068, 1.692 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.17294389402469892, negative=7, min=1.692, max=1.692, mean=0.20266666666666666, count=18.0, positive=11, stdDev=1.0660389819847638, zeros=0}
    Output: [
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, -2.9952 ], [ 0.0, 1.01376 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 0.51456 ], [ 0.0, 1.29024 ] ]
    ]
    Outputs Statistics: {meanExponent=0.07611687569627161, negative=1, min=1.29024, max=1.29024, mean=-0.00981333333333334, count=18.0, positive=3, stdDev=0.8139987101272881, zeros=14}
    Feedback for input 0
    Inputs Values: [
    	[ [ 0.132, -1.56 ], [ 1.816, 0.528 ], [ 1.604, -0.348 ] ],
    	[ [ 1.156, 0.268 ], [ 0.736, 0.672 ], [ -0.836, 0.448 ] ],
    	[ [ -1.916, -0.052 ], [ -0.656, 1.032 ], [ -1.068, 1.692 ] ]
    ]
    Value Statistics: {meanExponent=-0.17294389402469892, negative=7, min=1.692, 
```
...[skipping 3669 bytes](etc/39.txt)...
```
    , 0.0, 0.0, 0.0, 0.0, ... ], [ 2.220446049250313E-16, 0.0, -1.1102230246251565E-16, 2.220446049250313E-16, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 2.220446049250313E-16, 0.0, 0.0, 2.220446049250313E-16, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, -1.1102230246251565E-16, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 2.220446049250313E-16, 0.0, -1.1102230246251565E-16, 2.220446049250313E-16, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 2.220446049250313E-16, 0.0, 0.0, 2.220446049250313E-16, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.1102230246251565E-16, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.220446049250313E-16, 0.0, ... ], ... ]
    Error Statistics: {meanExponent=-14.059893927320354, negative=44, min=-6.602496327445806E-13, max=-6.602496327445806E-13, mean=2.1585913783451746E-14, count=648.0, positive=51, stdDev=2.708908469808987E-13, zeros=553}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 4.1411e-14 +- 2.2713e-13 [0.0000e+00 - 3.0305e-12] (972#)
    relativeTol: 2.0561e-13 +- 7.5279e-13 [0.0000e+00 - 6.9726e-12] (200#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=4.1411e-14 +- 2.2713e-13 [0.0000e+00 - 3.0305e-12] (972#), relativeTol=2.0561e-13 +- 7.5279e-13 [0.0000e+00 - 6.9726e-12] (200#)}
```



### Performance
Code from [StandardLayerTests.java:119](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L119) executed in 1.99 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[100, 100, 3]
    Performance:
    	Evaluation performance: 0.093770s +- 0.057366s [0.062916s - 0.208406s]
    	Learning performance: 0.234852s +- 0.018865s [0.222442s - 0.271960s]
    
```

