# ConvolutionLayer
## Double
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.ConvolutionLayer",
      "id": "cec22c4f-1794-4455-8e56-d68998c338c2",
      "isFrozen": false,
      "name": "ConvolutionLayer/cec22c4f-1794-4455-8e56-d68998c338c2",
      "filter": [
        [
          [
            -1.816
          ]
        ],
        [
          [
            -1.688
          ]
        ],
        [
          [
            1.808
          ]
        ],
        [
          [
            -0.208
          ]
        ]
      ],
      "strideX": 1,
      "strideY": 1
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
    	[ [ -0.304, -0.52 ], [ 0.044, 1.548 ], [ -1.8, 1.02 ], [ 0.832, -1.508 ], [ 0.688, 1.556 ] ],
    	[ [ 0.148, 0.948 ], [ 0.748, 0.86 ], [ -1.936, 0.7 ], [ 0.884, 1.144 ], [ -0.544, 0.94 ] ],
    	[ [ -0.436, 1.468 ], [ -1.8, -1.7 ], [ 0.264, -0.996 ], [ 1.8, 1.464 ], [ -0.836, -0.9 ] ],
    	[ [ 0.056, 1.728 ], [ -0.204, -0.248 ], [ 0.512, -1.584 ], [ -1.528, 0.848 ], [ 1.908, 1.396 ] ],
    	[ [ -0.824, 0.888 ], [ -0.056, 1.08 ], [ -0.948, 1.108 ], [ 1.0, -1.22 ], [ 0.984, 0.348 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -0.38809600000000005, 0.621312 ], [ 2.71888, -0.396256 ], [ 5.11296, 2.82624 ], [ -4.237376, -1.090752 ], [ 1.5638400000000003, -1.4849919999999999 ] ],
    	[ [ 1.445216, -0.44700799999999996 ], [ 0.196512, -1.441504 ], [ 4.781376, 3.122368 ], [ 0.4630079999999998, -1.730144 ], [ 2.687424, 0.7227520000000001 ] ],
    	[ [ 3.44592, 0.43062399999999995 ], [ 0.19520000000000015, 3.392 ], [ -2.280192, -0.23846400000000004 ], [ -0.6218880000000001, -3.3429119999999997 ], [ -0.10902400000000012, 1.598368 ] ],
    	[ [ 3.022528, -0.45395199999999997 ], [ -0.07791999999999999, 0.395936 ], [ -3.793664, -0.534784 ], [ 4.308032, 2.4028799999999997 ], [ -0.9409600000000001, -3.5110719999999995 ] ],
    	[ [ 3.101888, 1.206208 ], [ 2.054336, -0.130112 ], [ 3.724832, 1.3697599999999999 ], [ -4.0217600000000004, -1.43424 ], [ -1.1577600000000001, -1.733376 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ -3.504, 1.6 ], [ -3.504, 1.6 ], [ -3.504, 1.6 ], [ -3.504, 1.6 ], [ -3.504, 1.6 ] ],
    	[ [ -3.504, 1.6 ], [ -3.504, 1.6 ], [ -3.504, 1.6 ], [ -3.504, 1.6 ], [ -3.504, 1.6 ] ],
    	[ [ -3.504, 1.6 ], [ -3.504, 1.6 ], [ -3.504, 1.6 ], [ -3.504, 1.6 ], [ -3.504, 1.6 ] ],
    	[ [ -3.504, 1.6 ], [ -3.504, 1.6 ], [ -3.504, 1.6 ], [ -3.504, 1.6 ], [ -3.504, 1.6 ] ],
    	[ [ -3.504, 1.6 ], [ -3.504, 1.6 ], [ -3.504, 1.6 ], [ -3.504, 1.6 ], [ -3.504, 1.6 ] ]
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
      "id": "5ca1186e-78a3-4181-ad68-69bd15e0baae",
      "isFrozen": false,
      "name": "ConvolutionLayer/5ca1186e-78a3-4181-ad68-69bd15e0baae",
      "filter": [
        [
          [
            -1.816
          ]
        ],
        [
          [
            -1.688
          ]
        ],
        [
          [
            1.808
          ]
        ],
        [
          [
            -0.208
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
    	[ [ 1.148, 1.82 ], [ 1.716, -1.676 ], [ -0.712, -0.912 ], [ -1.432, -0.516 ], [ -0.612, 1.936 ] ],
    	[ [ -1.896, 0.492 ], [ -1.128, 1.968 ], [ -0.732, 1.644 ], [ 0.856, -0.872 ], [ 0.756, 0.764 ] ],
    	[ [ -0.188, -1.708 ], [ -0.184, 0.584 ], [ 0.136, 0.984 ], [ 0.556, 1.208 ], [ -1.788, 0.48 ] ],
    	[ [ 1.964, 1.044 ], [ -1.184, 1.192 ], [ -0.66, 0.82 ], [ 1.92, -0.864 ], [ -0.188, -1.612 ] ],
    	[ [ -1.988, 0.944 ], [ 1.892, 1.292 ], [ -1.264, 0.076 ], [ -0.916, 0.404 ], [ -1.872, 0.56 ] ]
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
Code from [StandardLayerTests.java:102](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L102) executed in 0.02 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (1000#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (1000#)}
```



### Differential Validation
Code from [StandardLayerTests.java:110](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L110) executed in 0.28 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -1.58, -0.336 ], [ 0.34, 1.668 ], [ 0.424, 1.908 ], [ 0.176, -0.38 ], [ -0.976, -0.004 ] ],
    	[ [ 0.776, -1.212 ], [ -0.436, -1.012 ], [ 0.204, -0.668 ], [ -0.896, -1.58 ], [ 1.192, 1.616 ] ],
    	[ [ -1.26, 0.848 ], [ -0.236, -1.768 ], [ 0.04, 1.624 ], [ 1.236, -0.972 ], [ -0.304, 0.656 ] ],
    	[ [ -1.092, 0.88 ], [ -0.776, -1.8 ], [ -1.284, -0.236 ], [ 1.628, 0.868 ], [ -0.612, 1.272 ] ],
    	[ [ -0.744, 0.632 ], [ 1.5, -1.672 ], [ -1.44, 0.852 ], [ -0.072, -0.16 ], [ 1.736, 0.408 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.18888755272296606, negative=27, min=0.408, max=0.408, mean=-0.020479999999999977, count=50.0, positive=23, stdDev=1.0731285522247556, zeros=0}
    Output: [
    	[ [ 2.2617920000000002, 2.7369280000000002 ], [ 2.398304, -0.9208639999999999 ], [ 2.67968, -1.1125759999999998 ], [ -1.006656, -0.21804799999999996 ], [ 1.765184, 1.6483199999999998 ] ],
    	[ [ -3.600512, -1.057792 ], [ -1.03792, 0.946464 ], [ -1.578208, -0.20540799999999998 ], [ -1.2295040000000002, 1.841088 ], [ 0.757056000000000
```
...[skipping 4716 bytes](etc/18.txt)...
```
    63, mean=-0.010240000000069643, count=200.0, positive=46, stdDev=0.7588855660767808, zeros=100}
    Gradient Error: [ [ 1.1954881529163686E-12, 1.099120794378905E-13, 1.5154544286133387E-12, -2.5353052990340075E-12, 8.102407633714392E-13, -1.3253287356462806E-12, -1.4352408150841711E-12, -1.2501111257279263E-13, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ -7.800982082528662E-13, 2.8956836928273333E-12, 2.180366998061345E-12, -1.3403722576299515E-12, 4.100053629940703E-13, -3.105737889086413E-12, -2.349231920106831E-13, 2.4507063045575705E-12, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ] ]
    Error Statistics: {meanExponent=-12.287299531128447, negative=50, min=-1.5903389716243055E-12, max=-1.5903389716243055E-12, mean=-6.963467996667916E-14, count=200.0, positive=50, stdDev=9.460929473343307E-13, zeros=100}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 7.2317e-14 +- 3.4754e-13 [0.0000e+00 - 5.8811e-12] (2700#)
    relativeTol: 8.9172e-13 +- 1.9009e-12 [7.7005e-15 - 1.4378e-11] (200#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=7.2317e-14 +- 3.4754e-13 [0.0000e+00 - 5.8811e-12] (2700#), relativeTol=8.9172e-13 +- 1.9009e-12 [7.7005e-15 - 1.4378e-11] (200#)}
```



### Performance
Code from [StandardLayerTests.java:120](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L120) executed in 0.02 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    Evaluation performance: 0.001168s +- 0.000322s [0.000781s - 0.001529s]
    Learning performance: 0.001051s +- 0.000082s [0.000950s - 0.001193s]
    
```

