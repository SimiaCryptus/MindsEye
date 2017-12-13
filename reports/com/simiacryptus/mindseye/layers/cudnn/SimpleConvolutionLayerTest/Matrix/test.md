# SimpleConvolutionLayer
## Matrix
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.SimpleConvolutionLayer",
      "id": "4faa5190-a67c-4136-8b19-f770daffb0d4",
      "isFrozen": false,
      "name": "SimpleConvolutionLayer/4faa5190-a67c-4136-8b19-f770daffb0d4",
      "filter": [
        [
          [
            1.608,
            -0.56,
            0.076
          ],
          [
            1.724,
            -0.62,
            1.568
          ],
          [
            0.604,
            0.692,
            -0.012
          ]
        ]
      ],
      "strideX": 1,
      "strideY": 1,
      "simple": false
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
    	[ [ 1.732 ], [ -0.164 ], [ 0.164 ] ],
    	[ [ 1.48 ], [ -1.548 ], [ -0.932 ] ],
    	[ [ 1.324 ], [ 1.28 ], [ -0.392 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -0.9196640000000004 ], [ -2.0651040000000003 ], [ -2.756928 ] ],
    	[ [ 6.993408 ], [ 4.616448 ], [ -0.13694399999999982 ] ],
    	[ [ 0.6653119999999999 ], [ -2.173728 ], [ -0.31400000000000017 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 1.6280000000000001 ], [ 1.144 ], [ 0.46399999999999997 ] ],
    	[ [ 3.9560000000000004 ], [ 5.08 ], [ 3.7960000000000003 ] ],
    	[ [ 2.4 ], [ 3.4479999999999995 ], [ 2.152 ] ]
    ]
```



[GPU Log](etc/cuda.log)

### Reference Implementation
Code from [StandardLayerTests.java:93](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L93) executed in 0.01 seconds: 
```java
    System.out.println(new GsonBuilder().setPrettyPrinting().create().toJson(referenceLayer.getJson()));
    getEquivalencyTester().test(referenceLayer, layer, inputPrototype);
```
Logging: 
```
    {
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "d4964fd7-288a-4c19-b8db-6cda1b51ddfc",
      "isFrozen": false,
      "name": "ConvolutionLayer/d4964fd7-288a-4c19-b8db-6cda1b51ddfc",
      "filter": [
        [
          [
            1.608,
            -0.56,
            0.076
          ],
          [
            1.724,
            -0.62,
            1.568
          ],
          [
            0.604,
            0.692,
            -0.012
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
    	[ [ -1.84 ], [ -0.88 ], [ 1.296 ] ],
    	[ [ 0.644 ], [ -0.868 ], [ 0.992 ] ],
    	[ [ -1.796 ], [ -1.128 ], [ -0.56 ] ]
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
Code from [StandardLayerTests.java:102](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L102) executed in 0.01 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (180#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (180#)}
```



### Differential Validation
Code from [StandardLayerTests.java:110](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L110) executed in 0.03 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 0.412 ], [ 0.772 ], [ 1.852 ] ],
    	[ [ 1.612 ], [ -1.096 ], [ 1.76 ] ],
    	[ [ 0.416 ], [ 0.74 ], [ -1.06 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.02483614735901693, negative=2, min=-1.06, max=-1.06, mean=0.6008888888888888, count=9.0, positive=7, stdDev=1.037499177450217, zeros=0}
    Output: [
    	[ [ 0.3289599999999999 ], [ 0.683568 ], [ 1.7582399999999998 ] ],
    	[ [ 2.226112 ], [ 1.9782719999999996 ], [ -0.3354400000000001 ] ],
    	[ [ 1.7720000000000002 ], [ -1.1814400000000003 ], [ 3.942112 ] ]
    ]
    Outputs Statistics: {meanExponent=0.07589502047378556, negative=2, min=3.942112, max=3.942112, mean=1.2413759999999998, count=9.0, positive=7, stdDev=1.4474285497511328, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 0.412 ], [ 0.772 ], [ 1.852 ] ],
    	[ [ 1.612 ], [ -1.096 ], [ 1.76 ] ],
    	[ [ 0.416 ], [ 0.74 ], [ -1.06 ] ]
    ]
    Value Statistics: {meanExponent=-0.02483614735901693, negative=2, min=-1.06, max=-1.06, mean=0.6008888888888888, count=9.0, positive=7, stdDev=1.037499177450217, zeros=0}
    Imple
```
...[skipping 6782 bytes](etc/51.txt)...
```
    , 1.7601475832407232E-12, 2.8257396422759484E-12 ], [ 0.0, -3.695765915523452E-12, -2.753353101070388E-13, 0.0, -2.225553075163589E-12, 1.7905676941154525E-12, 0.0, -3.530509218307998E-14, 1.7601475832407232E-12 ], [ 0.0, 0.0, 0.0, -2.753353101070388E-13, 8.6014528832834E-13, 0.0, 1.7905676941154525E-12, 1.8496315590255108E-13, 0.0 ], [ 0.0, 0.0, 0.0, -3.6509684164798273E-13, -2.753353101070388E-13, 8.6014528832834E-13, -5.10702591327572E-15, 1.2523315717771766E-13, -3.1457059179729185E-12 ], [ 0.0, 0.0, 0.0, 0.0, 7.451261829771738E-13, -2.753353101070388E-13, 0.0, -5.10702591327572E-15, -4.298783551348606E-13 ] ]
    Error Statistics: {meanExponent=-12.235052400459026, negative=25, min=-4.298783551348606E-13, max=-4.298783551348606E-13, mean=1.6925829736408806E-13, count=81.0, positive=24, stdDev=1.2106341447207033E-12, zeros=32}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 6.5937e-13 +- 9.4793e-13 [0.0000e+00 - 3.7610e-12] (162#)
    relativeTol: 1.7911e-12 +- 8.0627e-12 [3.3077e-15 - 7.8141e-11] (98#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=6.5937e-13 +- 9.4793e-13 [0.0000e+00 - 3.7610e-12] (162#), relativeTol=1.7911e-12 +- 8.0627e-12 [3.3077e-15 - 7.8141e-11] (98#)}
```



### Performance
Code from [StandardLayerTests.java:120](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L120) executed in 0.01 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    Evaluation performance: 0.000463s +- 0.000091s [0.000369s - 0.000622s]
    Learning performance: 0.000458s +- 0.000022s [0.000434s - 0.000496s]
    
```

