# SimpleConvolutionLayer
## MultiBand
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
      "id": "ce9f9780-4345-4383-83c4-689509f87756",
      "isFrozen": false,
      "name": "SimpleConvolutionLayer/ce9f9780-4345-4383-83c4-689509f87756",
      "filter": [
        [
          [
            1.188
          ]
        ],
        [
          [
            0.796
          ]
        ],
        [
          [
            1.848
          ]
        ],
        [
          [
            0.148
          ]
        ],
        [
          [
            0.368
          ]
        ],
        [
          [
            0.668
          ]
        ],
        [
          [
            1.152
          ]
        ],
        [
          [
            0.2
          ]
        ],
        [
          [
            1.012
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
    	[ [ 0.468, 0.028, 0.608 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.701856, 0.48571200000000003, 1.160032 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 2.4879999999999995, 1.364, 3.528 ] ]
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
      "id": "5423554e-fa23-47a2-9d27-c511c2706b38",
      "isFrozen": false,
      "name": "ConvolutionLayer/5423554e-fa23-47a2-9d27-c511c2706b38",
      "filter": [
        [
          [
            1.188
          ]
        ],
        [
          [
            0.148
          ]
        ],
        [
          [
            1.152
          ]
        ],
        [
          [
            0.796
          ]
        ],
        [
          [
            0.368
          ]
        ],
        [
          [
            0.2
          ]
        ],
        [
          [
            1.848
          ]
        ],
        [
          [
            0.668
          ]
        ],
        [
          [
            1.012
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
    	[ [ -1.044, -0.68, 1.352 ] ]
    ]
    Error: [
    	[ [ 0.0, 0.0, 0.0 ] ]
    ]
    Accuracy:
    absoluteTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (3#)
    relativeTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (3#)
    
```

### Batch Execution
Code from [StandardLayerTests.java:102](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L102) executed in 0.01 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (60#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (60#)}
```



### Differential Validation
Code from [StandardLayerTests.java:110](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L110) executed in 0.02 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -1.296, -1.032, 0.452 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.0728589554542836, negative=2, min=0.452, max=0.452, mean=-0.6253333333333334, count=3.0, positive=1, stdDev=0.7693760820012249, zeros=0}
    Output: [
    	[ [ -1.525824, -0.26964799999999994, -1.2419679999999997 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.09719598289817853, negative=3, min=-1.2419679999999997, max=-1.2419679999999997, mean=-1.0124799999999998, count=3.0, positive=0, stdDev=0.5378928594382589, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -1.296, -1.032, 0.452 ] ]
    ]
    Value Statistics: {meanExponent=-0.0728589554542836, negative=2, min=0.452, max=0.452, mean=-0.6253333333333334, count=3.0, positive=1, stdDev=0.7693760820012249, zeros=0}
    Implemented Feedback: [ [ 1.188, 0.148, 1.152 ], [ 0.796, 0.368, 0.2 ], [ 1.848, 0.668, 1.012 ] ]
    Implemented Statistics: {meanExponent=-0.20322439343712143, negative=0, min=1.012, max=1.012, mean=0.8200000000000001, count=9.0, positive=9, stdDev=0.5167488106743289, zeros=0}
    Mea
```
...[skipping 1650 bytes](etc/52.txt)...
```
    0, 0.0, -1.03200000000081 ], [ 0.0, 0.0, 0.4519999999996749 ] ]
    Measured Statistics: {meanExponent=-0.0728589554545248, negative=6, min=0.4519999999996749, max=0.4519999999996749, mean=-0.20844444444435273, count=27.0, positive=3, stdDev=0.5331147700302283, zeros=18}
    Gradient Error: [ [ 2.700728529703156E-12, 0.0, 0.0 ], [ 1.4104273304837989E-12, 0.0, 0.0 ], [ -3.251288127614771E-13, 0.0, 0.0 ], [ 0.0, -6.299405441723138E-13, 0.0 ], [ 0.0, 3.0020430585864233E-13, 0.0 ], [ 0.0, -3.251288127614771E-13, 0.0 ], [ 0.0, 0.0, 4.802824804528427E-13 ], [ 0.0, 0.0, -8.100187187665142E-13 ], [ 0.0, 0.0, -3.251288127614771E-13 ] ]
    Error Statistics: {meanExponent=-12.224032633331671, negative=5, min=-3.251288127614771E-13, max=-3.251288127614771E-13, mean=9.171470167685854E-14, count=27.0, positive=4, stdDev=6.3090183966852E-13, zeros=18}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 3.7412e-13 +- 6.7495e-13 [0.0000e+00 - 2.9055e-12] (36#)
    relativeTol: 5.5347e-13 +- 5.9030e-13 [4.0672e-14 - 2.2755e-12] (18#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=3.7412e-13 +- 6.7495e-13 [0.0000e+00 - 2.9055e-12] (36#), relativeTol=5.5347e-13 +- 5.9030e-13 [4.0672e-14 - 2.2755e-12] (18#)}
```



### Performance
Code from [StandardLayerTests.java:120](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L120) executed in 0.01 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    Evaluation performance: 0.000419s +- 0.000043s [0.000379s - 0.000474s]
    Learning performance: 0.000447s +- 0.000034s [0.000408s - 0.000491s]
    
```

