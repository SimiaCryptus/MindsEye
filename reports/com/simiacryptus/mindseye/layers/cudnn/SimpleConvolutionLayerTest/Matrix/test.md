# SimpleConvolutionLayer
## Matrix
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
      "id": "762a18fd-c70c-486b-9278-6d0d2f8b6160",
      "isFrozen": false,
      "name": "SimpleConvolutionLayer/762a18fd-c70c-486b-9278-6d0d2f8b6160",
      "filter": [
        [
          [
            0.624,
            1.92,
            -0.46
          ],
          [
            -1.212,
            -1.616,
            1.608
          ],
          [
            -1.94,
            -0.672,
            -1.844
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
    	[ [ -1.868 ], [ -1.584 ], [ 1.084 ] ],
    	[ [ 0.888 ], [ -0.384 ], [ 0.116 ] ],
    	[ [ -1.464 ], [ -0.088 ], [ -0.104 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -1.3384639999999999 ], [ 4.711392000000001 ], [ -0.08292800000000006 ] ],
    	[ [ -2.7279360000000006 ], [ 3.527328 ], [ 5.031328 ] ],
    	[ [ 3.8014080000000003 ], [ -1.381968 ], [ 1.1218240000000002 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ -2.524 ], [ -1.064 ], [ 1.452 ] ],
    	[ [ -5.675999999999999 ], [ -3.5920000000000005 ], [ 0.8639999999999997 ] ],
    	[ [ -5.4399999999999995 ], [ -2.896 ], [ -0.28400000000000014 ] ]
    ]
```



[GPU Log](etc/cuda.log)

### Reference Implementation
Code from [StandardLayerTests.java:92](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L92) executed in 0.02 seconds: 
```java
    System.out.println(new GsonBuilder().setPrettyPrinting().create().toJson(referenceLayer.getJson()));
    getEquivalencyTester().test(referenceLayer, layer, inputPrototype);
```
Logging: 
```
    {
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "d1a4971e-ad97-485e-ae0b-f8a193c26e1e",
      "isFrozen": false,
      "name": "ConvolutionLayer/d1a4971e-ad97-485e-ae0b-f8a193c26e1e",
      "filter": [
        [
          [
            0.624,
            1.92,
            -0.46
          ],
          [
            -1.212,
            -1.616,
            1.608
          ],
          [
            -1.94,
            -0.672,
            -1.844
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
    	[ [ -0.248 ], [ -0.78 ], [ 1.196 ] ],
    	[ [ 0.16 ], [ 0.148 ], [ 1.596 ] ],
    	[ [ 1.98 ], [ 1.512 ], [ -1.904 ] ]
    ]
    Error: [
    	[ [ 0.0 ], [ 0.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.0 ], [ 0.0 ] ]
    ]
    Accuracy:
    absoluteTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (9#)
    relativeTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (9#)
    
```

### Batch Execution
Code from [StandardLayerTests.java:101](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L101) executed in 0.02 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.4803e-17 +- 7.9716e-17 [0.0000e+00 - 4.4409e-16] (180#), relativeTol=8.5665e-18 +- 4.6132e-17 [0.0000e+00 - 2.5700e-16] (180#)}
```



### Differential Validation
Code from [StandardLayerTests.java:109](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L109) executed in 0.06 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 1.028 ], [ 1.956 ], [ -0.096 ] ],
    	[ [ 0.948 ], [ -1.552 ], [ 0.144 ] ],
    	[ [ 1.536 ], [ -0.632 ], [ -1.608 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.13276776856708844, negative=4, min=-1.608, max=-1.608, mean=0.19155555555555553, count=9.0, positive=5, stdDev=1.2126290182089778, zeros=0}
    Output: [
    	[ [ -0.02315200000000037 ], [ -3.9042719999999997 ], [ 1.6770559999999999 ] ],
    	[ [ -6.014544 ], [ 0.2239840000000005 ], [ 0.22398399999999993 ] ],
    	[ [ -1.4573120000000004 ], [ -7.408208 ], [ 6.116672000000001 ] ]
    ]
    Outputs Statistics: {meanExponent=0.05334374939194833, negative=5, min=6.116672000000001, max=6.116672000000001, mean=-1.1739768888888886, count=9.0, positive=4, stdDev=3.892249487493731, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 1.028 ], [ 1.956 ], [ -0.096 ] ],
    	[ [ 0.948 ], [ -1.552 ], [ 0.144 ] ],
    	[ [ 1.536 ], [ -0.632 ], [ -1.608 ] ]
    ]
    Value Statistics: {meanExponent=-0.13276776856708844, negative=4, min=-1.608, max=-1.608, mean=0.19155555555555553, count=9.0, posi
```
...[skipping 6932 bytes](etc/89.txt)...
```
    9E-12, -3.000655279805642E-13, -6.270983732292734E-12 ], [ 0.0, -3.745892485085278E-12, 6.150635556423367E-13, 0.0, -3.7059244561987725E-12, -2.19824158875781E-13, 0.0, 1.68035030334579E-12, -4.74095762648119E-12 ], [ 0.0, 0.0, 0.0, -1.6053824936079764E-12, -2.402522625288839E-13, 0.0, -2.19824158875781E-13, -4.100053629940703E-13, 0.0 ], [ 0.0, 0.0, 0.0, -3.745892485085278E-12, -1.6053824936079764E-12, -2.402522625288839E-13, 7.349676423018536E-13, -2.19824158875781E-13, -4.8508974614946965E-12 ], [ 0.0, 0.0, 0.0, 0.0, -1.376676550535194E-13, 2.83550960489265E-12, 0.0, -9.769962616701378E-14, -2.19824158875781E-13 ] ]
    Error Statistics: {meanExponent=-12.032787986168241, negative=33, min=-2.19824158875781E-13, max=-2.19824158875781E-13, mean=-2.6692708400264E-13, count=81.0, positive=16, stdDev=2.051208515048566E-12, zeros=32}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.0902e-12 +- 1.5031e-12 [0.0000e+00 - 6.9615e-12] (162#)
    relativeTol: 1.6717e-12 +- 3.2665e-12 [3.3114e-15 - 1.6462e-11] (98#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.0902e-12 +- 1.5031e-12 [0.0000e+00 - 6.9615e-12] (162#), relativeTol=1.6717e-12 +- 3.2665e-12 [3.3114e-15 - 1.6462e-11] (98#)}
```



### Performance
Code from [StandardLayerTests.java:119](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L119) executed in 0.40 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[100, 100, 1]
    Performance:
    	Evaluation performance: 0.014632s +- 0.003550s [0.010224s - 0.018101s]
    	Learning performance: 0.050385s +- 0.015389s [0.036415s - 0.069548s]
    
```

