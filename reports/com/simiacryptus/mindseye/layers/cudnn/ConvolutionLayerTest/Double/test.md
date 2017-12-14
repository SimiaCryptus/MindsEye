# ConvolutionLayer
## Double
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.ConvolutionLayer",
      "id": "d52b44cf-45e5-4665-b695-600535a8ff7a",
      "isFrozen": false,
      "name": "ConvolutionLayer/d52b44cf-45e5-4665-b695-600535a8ff7a",
      "filter": [
        [
          [
            -1.892
          ]
        ],
        [
          [
            0.096
          ]
        ],
        [
          [
            1.912
          ]
        ],
        [
          [
            -0.044
          ]
        ]
      ],
      "strideX": 1,
      "strideY": 1
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
    	[ [ -1.944, 0.476 ], [ 0.648, -1.108 ], [ -1.972, -1.424 ], [ 0.332, 0.74 ], [ -0.668, -1.244 ] ],
    	[ [ 0.16, 1.316 ], [ -0.576, -0.116 ], [ 0.12, 1.052 ], [ 1.644, -0.344 ], [ 1.916, 0.84 ] ],
    	[ [ 1.28, -0.048 ], [ -0.508, 1.148 ], [ 0.092, 0.664 ], [ -0.464, 1.12 ], [ 0.52, 1.436 ] ],
    	[ [ 0.292, 1.268 ], [ 0.944, 1.108 ], [ 1.22, 1.7 ], [ 1.272, 1.156 ], [ 0.108, -1.236 ] ],
    	[ [ 1.172, 0.408 ], [ -0.024, -0.252 ], [ -1.68, 1.86 ], [ -1.336, 0.468 ], [ 0.664, -0.372 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 4.588159999999999, -0.207568 ], [ -3.344512, 0.11096 ], [ 1.008336, -0.12665600000000002 ], [ 0.7867359999999999, -6.879999999999932E-4 ], [ -1.1146719999999999, -0.009392000000000008 ] ],
    	[ [ 2.213472, -0.042544 ], [ 0.8679999999999999, -0.050192 ], [ 1.784384, -0.034768 ], [ -3.768176, 0.17296 ], [ -2.018992, 0.146976 ] ],
    	[ [ -2.5135359999999998, 0.124992 ], [ 3.156112, -0.09928 ], [ 1.095504, -0.020384 ], [ 3.0193280000000002, -0.093824 ], [ 1.7617919999999998, 
```
...[skipping 541 bytes](etc/53.txt)...
```
    99999999 ], [ -1.7959999999999998, 1.8679999999999999 ], [ -1.7959999999999998, 1.8679999999999999 ] ],
    	[ [ -1.7959999999999998, 1.8679999999999999 ], [ -1.7959999999999998, 1.8679999999999999 ], [ -1.7959999999999998, 1.8679999999999999 ], [ -1.7959999999999998, 1.8679999999999999 ], [ -1.7959999999999998, 1.8679999999999999 ] ],
    	[ [ -1.7959999999999998, 1.8679999999999999 ], [ -1.7959999999999998, 1.8679999999999999 ], [ -1.7959999999999998, 1.8679999999999999 ], [ -1.7959999999999998, 1.8679999999999999 ], [ -1.7959999999999998, 1.8679999999999999 ] ],
    	[ [ -1.7959999999999998, 1.8679999999999999 ], [ -1.7959999999999998, 1.8679999999999999 ], [ -1.7959999999999998, 1.8679999999999999 ], [ -1.7959999999999998, 1.8679999999999999 ], [ -1.7959999999999998, 1.8679999999999999 ] ],
    	[ [ -1.7959999999999998, 1.8679999999999999 ], [ -1.7959999999999998, 1.8679999999999999 ], [ -1.7959999999999998, 1.8679999999999999 ], [ -1.7959999999999998, 1.8679999999999999 ], [ -1.7959999999999998, 1.8679999999999999 ] ]
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
      "id": "a9a6acec-e5bf-4502-ac76-f8777d1113a5",
      "isFrozen": false,
      "name": "ConvolutionLayer/a9a6acec-e5bf-4502-ac76-f8777d1113a5",
      "filter": [
        [
          [
            -1.892
          ]
        ],
        [
          [
            0.096
          ]
        ],
        [
          [
            1.912
          ]
        ],
        [
          [
            -0.044
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
    	[ [ -1.108, -0.296 ], [ -1.264, -0.032 ], [ -0.192, -0.736 ], [ -1.5, -1.832 ], [ 0.772, -0.708 ] ],
    	[ [ 1.992, 1.576 ], [ -1.564, -0.456 ], [ 0.788, -1.3 ], [ 0.764, -0.108 ], [ 1.884, -0.328 ] ],
    	[ [ -0.232, -1.56 ], [ 1.772, -1.1 ], [ -0.5, -1.708 ], [ 1.652, 1.108 ], [ 1.904, -1.32 ] ],
    	[ [ -0.252, 1.996 ], [ 0.76, 0.704 ], [ -1.388, 1.884 ], [ 0.332, 1.228 ], [ 0.976, -0.148 ] ],
    	[ [ -0.592, 1.604 ], [ 1.032, 0.232 ], [ -0.456, 0.576 ], [ -1.432, 0.844 ], [ 0.032, -1.304 ] ]
    ]
    Error: [
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ]
    ]
    Accuracy:
    absoluteTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (50#)
    relativeTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (50#)
    
```

### Batch Execution
Code from [StandardLayerTests.java:101](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L101) executed in 0.02 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (1000#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (1000#)}
```



### Differential Validation
Code from [StandardLayerTests.java:109](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L109) executed in 0.19 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -1.36, -1.84 ], [ -0.252, -0.536 ], [ -0.392, -1.204 ], [ 0.148, 0.8 ], [ -1.224, 1.78 ] ],
    	[ [ 1.296, 1.088 ], [ -0.868, -0.556 ], [ -1.108, -0.396 ], [ -1.484, -1.192 ], [ -1.916, 1.064 ] ],
    	[ [ 0.372, -0.58 ], [ 0.392, -1.744 ], [ -1.44, -0.464 ], [ -1.964, 1.916 ], [ 0.652, 1.66 ] ],
    	[ [ 1.16, -0.276 ], [ -0.84, -1.548 ], [ -0.08, 0.54 ], [ 1.428, 0.948 ], [ 1.032, -1.36 ] ],
    	[ [ -0.796, -0.02 ], [ 0.06, 1.576 ], [ -0.848, -0.048 ], [ 0.716, -0.4 ], [ -1.16, 1.048 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.15774377747923857, negative=30, min=1.048, max=1.048, mean=-0.16440000000000005, count=50.0, positive=20, stdDev=1.0881594736066953, zeros=0}
    Output: [
    	[ [ -0.9449600000000001, -0.04960000000000001 ], [ -0.548048, -6.080000000000017E-4 ], [ -1.5603839999999998, 0.015343999999999997 ], [ 1.249584, -0.020992 ], [ 5.719168, -0.195824 ] ],
    	[ [ -0.37177599999999994, 0.07654400000000001 ], [ 0.5791839999999999, -0.058864 ], [ 1.339184, -0.08894400000000002 ], [ 0.5286240000000002, -0.090
```
...[skipping 4731 bytes](etc/54.txt)...
```
    0000000016, mean=-0.08219999999994215, count=200.0, positive=40, stdDev=0.7738232097837734, zeros=100}
    Gradient Error: [ [ 8.604228440844963E-13, 1.7401635687974704E-12, 1.8153256675645935E-12, 2.270406085358445E-12, -6.850076061937216E-13, -5.850875339774575E-13, -5.350164755668629E-13, -2.0504709041802016E-12, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 3.8036240823657863E-13, 1.9984014443252818E-13, 1.0852430065710905E-12, 3.1656899324161714E-12, -5.751128739905909E-13, -9.801048861390882E-13, -4.4497738826976274E-13, 3.140820936664568E-12, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ] ]
    Error Statistics: {meanExponent=-12.8130059292638, negative=49, min=1.5987211554602254E-13, max=1.5987211554602254E-13, mean=5.783856033003688E-14, count=200.0, positive=51, stdDev=9.388726612195365E-13, zeros=100}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 5.7079e-14 +- 4.0565e-13 [0.0000e+00 - 7.4714e-12] (2700#)
    relativeTol: 5.2766e-13 +- 1.3431e-12 [5.5884e-16 - 1.4378e-11] (200#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=5.7079e-14 +- 4.0565e-13 [0.0000e+00 - 7.4714e-12] (2700#), relativeTol=5.2766e-13 +- 1.3431e-12 [5.5884e-16 - 1.4378e-11] (200#)}
```



### Performance
Code from [StandardLayerTests.java:119](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L119) executed in 0.52 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[100, 100, 2]
    Performance:
    	Evaluation performance: 0.015208s +- 0.000872s [0.014079s - 0.016537s]
    	Learning performance: 0.072368s +- 0.015731s [0.058184s - 0.093784s]
    
```

