# SimpleConvolutionLayer
## Matrix
### Json Serialization
Code from [LayerTestBase.java:83](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L83) executed in 0.00 seconds: 
```java
    JsonObject json = layer.getJson();
    NNLayer echo = NNLayer.fromJson(json);
    assert (echo != null) : "Failed to deserialize";
    assert (layer != echo) : "Serialization did not copy";
    Assert.assertEquals("Serialization not equal", layer, echo);
    return new GsonBuilder().setPrettyPrinting().create().toJson(json);
```

Returns: 

```
    {
      "class": "com.simiacryptus.mindseye.layers.cudnn.f64.SimpleConvolutionLayer",
      "id": "370a9587-74a1-4959-b406-fa4500002b64",
      "isFrozen": false,
      "name": "SimpleConvolutionLayer/370a9587-74a1-4959-b406-fa4500002b64",
      "filter": {
        "dimensions": [
          3,
          3,
          1
        ],
        "data": [
          0.836,
          1.212,
          -0.544,
          -1.652,
          1.188,
          1.928,
          -1.668,
          -0.028,
          -0.972
        ]
      },
      "strideX": 1,
      "strideY": 1,
      "simple": false
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:120](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L120) executed in 0.00 seconds: 
```java
    SimpleEval eval = SimpleEval.run(layer, inputPrototype);
    return String.format("--------------------\nInput: \n[%s]\n--------------------\nOutput: \n%s",
      Arrays.stream(inputPrototype).map(t->t.prettyPrint()).reduce((a,b)->a+",\n"+b).get(),
      eval.getOutput().prettyPrint());
```

Returns: 

```
    --------------------
    Input: 
    [[
    	[ [ 0.776 ], [ -1.784 ], [ -1.192 ] ],
    	[ [ 1.664 ], [ -1.384 ], [ -1.704 ] ],
    	[ [ 0.66 ], [ -0.944 ], [ -0.144 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -5.146272 ], [ -5.4995519999999996 ], [ 3.757376 ] ],
    	[ [ 0.886544 ], [ -6.963184000000001 ], [ -0.7372479999999992 ] ],
    	[ [ 3.60104 ], [ -4.673264 ], [ -2.0847040000000003 ] ]
    ]
```



### Reference Implementation
Code from [LayerTestBase.java:131](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L131) executed in 0.01 seconds: 
```java
    System.out.println(new GsonBuilder().setPrettyPrinting().create().toJson(referenceLayer.getJson()));
    getEquivalencyTester().test(referenceLayer, layer, inputPrototype);
```
Logging: 
```
    {
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "370a9587-74a1-4959-b406-fa4500002b65",
      "isFrozen": false,
      "name": "ConvolutionLayer/370a9587-74a1-4959-b406-fa4500002b65",
      "filter": {
        "dimensions": [
          3,
          3,
          1
        ],
        "data": [
          0.836,
          1.212,
          -0.544,
          -1.652,
          1.188,
          1.928,
          -1.668,
          -0.028,
          -0.972
        ]
      },
      "skip": {
        "dimensions": [
          1,
          1
        ]
      },
      "simple": true
    }
    Inputs: [
    	[ [ 0.776 ], [ -1.784 ], [ -1.192 ] ],
    	[ [ 1.664 ], [ -1.384 ], [ -1.704 ] ],
    	[ [ 0.66 ], [ -0.944 ], [ -0.144 ] ]
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
Code from [LayerTestBase.java:138](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.01 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=3.8549e-17 +- 1.6351e-16 [0.0000e+00 - 8.8818e-16] (180#), relativeTol=2.0756e-17 +- 7.6237e-17 [0.0000e+00 - 3.7007e-16] (180#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.04 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 0.776 ], [ -1.784 ], [ -1.192 ] ],
    	[ [ 1.664 ], [ -1.384 ], [ -1.704 ] ],
    	[ [ 0.66 ], [ -0.944 ], [ -0.144 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.026203305428464115, negative=6, min=-0.144, max=-0.144, mean=-0.4502222222222222, count=9.0, positive=3, stdDev=1.1698449828719262, zeros=0}
    Output: [
    	[ [ -5.146272 ], [ -5.4995519999999996 ], [ 3.757376 ] ],
    	[ [ 0.886544 ], [ -6.963184000000001 ], [ -0.7372479999999992 ] ],
    	[ [ 3.60104 ], [ -4.673264 ], [ -2.0847040000000003 ] ]
    ]
    Outputs Statistics: {meanExponent=0.4699910151888631, negative=6, min=-2.0847040000000003, max=-2.0847040000000003, mean=-1.8732515555555556, count=9.0, positive=3, stdDev=3.7746776484702815, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 0.776 ], [ -1.784 ], [ -1.192 ] ],
    	[ [ 1.664 ], [ -1.384 ], [ -1.704 ] ],
    	[ [ 0.66 ], [ -0.944 ], [ -0.144 ] ]
    ]
    Value Statistics: {meanExponent=-0.026203305428464115, negative=6, min=-0.144, max=-0.144, mean=-0.4502222222222222, count=9.0, positive=3, stdDev=1.1698449828719262, zeros=0}
    Implemented Feedback: [ [ 1.188, 1.928, 0.0, -0.028, -0.972, 0.0, 0.0, 0.0, 0.0 ], [ -1.652, 1.188, 1.928, -1.668, -0.028, -0.972, 0.0, 0.0, 0.0 ], [ 0.0, -1.652, 1.188, 0.0, -1.668, -0.028, 0.0, 0.0, 0.0 ], [ 1.212, -0.544, 0.0, 1.188, 1.928, 0.0, -0.028, -0.972, 0.0 ], [ 0.836, 1.212, -0.544, -1.652, 1.188, 1.928, -1.668, -0.028, -0.972 ], [ 0.0, 0.836, 1.212, 0.0, -1.652, 1.188, 0.0, -1.668, -0.028 ], [ 0.0, 0.0, 0.0, 1.212, -0.544, 0.0, 1.188, 1.928, 0.0 ], [ 0.0, 0.0, 0.0, 0.836, 1.212, -0.544, -1.652, 1.188, 1.928 ], [ 0.0, 0.0, 0.0, 0.0, 0.836, 1.212, 0.0, -1.652, 1.188 ] ]
    Implemented Statistics: {meanExponent=-0.11537379057940188, negative=24, min=1.188, max=1.188, mean=0.12419753086419756, count=81.0, positive=25, stdDev=0.9802753760211513, zeros=32}
    Measured Feedback: [ [ 1.188000000000855, 1.9279999999999298, 0.0, -0.027999999998584713, -0.9719999999990847, 0.0, 0.0, 0.0, 0.0 ], [ -1.651999999996434, 1.1879999999997448, 1.9280000000021502, -1.668000000005776, -0.027999999998
```
...[skipping 4853 bytes](etc/1.txt)...
```
    034405, 0.0, -1.7840000000002298, -1.3839999999998298 ] ]
    Measured Statistics: {meanExponent=0.022864101400311337, negative=35, min=-1.3839999999998298, max=-1.3839999999998298, mean=-0.35387654320948486, count=81.0, positive=14, stdDev=0.9653945318069901, zeros=32}
    Gradient Error: [ [ 1.7008616737257398E-13, -5.000444502911705E-13, 0.0, -1.4988010832439613E-13, 4.74095762648119E-12, 0.0, 0.0, 0.0, 0.0 ], [ 6.431521981653532E-12, 1.7008616737257398E-13, -5.000444502911705E-13, -3.190558928167775E-12, -1.4988010832439613E-13, 4.74095762648119E-12, 0.0, 0.0, 0.0 ], [ 0.0, -2.4502622153477205E-12, 1.7008616737257398E-13, 0.0, 5.6912252688334775E-12, -1.4988010832439613E-13, 0.0, 0.0, 0.0 ], [ 3.44058115331336E-12, -1.0053069487980792E-12, 0.0, 1.7008616737257398E-13, -5.000444502911705E-13, 0.0, -1.4988010832439613E-13, -4.140826570520062E-12, 0.0 ], [ 1.099120794378905E-13, 1.099120794378905E-13, 3.435585149702547E-12, -2.4502622153477205E-12, 1.7008616737257398E-13, -5.000444502911705E-13, 1.2503331703328513E-12, -1.4988010832439613E-13, 3.000655279805642E-13 ], [ 0.0, 1.099120794378905E-13, 3.44058115331336E-12, 0.0, 6.431521981653532E-12, 1.7008616737257398E-13, 0.0, -3.190558928167775E-12, -1.4988010832439613E-13 ], [ 0.0, 0.0, 0.0, -5.441203043687892E-12, 3.435585149702547E-12, 0.0, 1.7008616737257398E-13, -5.000444502911705E-13, 0.0 ], [ 0.0, 0.0, 0.0, 1.099120794378905E-13, 3.44058115331336E-12, 3.435585149702547E-12, -2.4502622153477205E-12, 1.7008616737257398E-13, 3.940847648209456E-12 ], [ 0.0, 0.0, 0.0, 0.0, 1.099120794378905E-13, 3.44058115331336E-12, 0.0, -2.298161660974074E-13, 1.7008616737257398E-13 ] ]
    Error Statistics: {meanExponent=-12.155703797796756, negative=20, min=1.7008616737257398E-13, max=1.7008616737257398E-13, mean=3.9170724277339644E-13, count=81.0, positive=29, stdDev=2.0351266855110504E-12, zeros=32}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.2358e-12 +- 1.8557e-12 [0.0000e+00 - 8.0267e-12] (162#)
    relativeTol: 3.3602e-12 +- 9.1255e-12 [1.8197e-14 - 5.4029e-11] (98#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.2358e-12 +- 1.8557e-12 [0.0000e+00 - 8.0267e-12] (162#), relativeTol=3.3602e-12 +- 9.1255e-12 [1.8197e-14 - 5.4029e-11] (98#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.15 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 5.7113 +- 0.5937 [5.1068 - 8.2017]
    Learning performance: 3.5258 +- 0.2457 [3.2174 - 4.7221]
    
```

