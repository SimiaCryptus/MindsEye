# SimpleConvolutionLayer
## MultiBand
### Json Serialization
Code from [LayerTestBase.java:83](../../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L83) executed in 0.00 seconds: 
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
      "id": "a864e734-2f23-44db-97c1-504000002b6f",
      "isFrozen": false,
      "name": "SimpleConvolutionLayer/a864e734-2f23-44db-97c1-504000002b6f",
      "filter": {
        "dimensions": [
          1,
          1,
          9
        ],
        "data": [
          0.168,
          0.592,
          -1.424,
          -1.868,
          1.252,
          1.144,
          -0.664,
          -1.36,
          -1.512
        ]
      },
      "strideX": 1,
      "strideY": 1,
      "simple": false
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:120](../../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L120) executed in 0.00 seconds: 
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
    	[ [ 0.364, -0.9, 1.048 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -1.964, -0.6078400000000002, -0.6022719999999999 ] ]
    ]
```



### Reference Implementation
Code from [LayerTestBase.java:131](../../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L131) executed in 0.01 seconds: 
```java
    System.out.println(new GsonBuilder().setPrettyPrinting().create().toJson(referenceLayer.getJson()));
    getEquivalencyTester().test(referenceLayer, layer, inputPrototype);
```
Logging: 
```
    {
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "a864e734-2f23-44db-97c1-504000002b70",
      "isFrozen": false,
      "name": "ConvolutionLayer/a864e734-2f23-44db-97c1-504000002b70",
      "filter": {
        "dimensions": [
          1,
          1,
          9
        ],
        "data": [
          0.168,
          -1.868,
          -0.664,
          0.592,
          1.252,
          -1.36,
          -1.424,
          1.144,
          -1.512
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
    	[ [ 0.364, -0.9, 1.048 ] ]
    ]
    Error: [
    	[ [ 0.0, 0.0, 0.0 ] ]
    ]
    Accuracy:
    absoluteTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (3#)
    relativeTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (3#)
    
```

### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.02 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (60#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (60#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.02 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 0.364, -0.9, 1.048 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.15476494142130376, negative=1, min=1.048, max=1.048, mean=0.17066666666666666, count=3.0, positive=2, stdDev=0.80693218770573, zeros=0}
    Output: [
    	[ [ -1.964, -0.6078400000000002, -0.6022719999999999 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.047758855773064234, negative=3, min=-0.6022719999999999, max=-0.6022719999999999, mean=-1.0580373333333333, count=3.0, positive=0, stdDev=0.6406163780304783, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 0.364, -0.9, 1.048 ] ]
    ]
    Value Statistics: {meanExponent=-0.15476494142130376, negative=1, min=1.048, max=1.048, mean=0.17066666666666666, count=3.0, positive=2, stdDev=0.80693218770573, zeros=0}
    Implemented Feedback: [ [ 0.168, -1.868, -0.664 ], [ 0.592, 1.252, -1.36 ], [ -1.424, 1.144, -1.512 ] ]
    Implemented Statistics: {meanExponent=-0.03179922423571847, negative=5, min=-1.512, max=-1.512, mean=-0.40800000000000003, count=9.0, positive=4, stdDev=1.1474896271620256, zeros=0}
    Measured Feedback: [ [ 0.16800000000039006, -1.8680000000004249, -0.66400000000022 ], [ 0.5919999999992598, 1.252000000000475, -1.36000000000025 ], [ -1.4239999999987596, 1.1439999999995898, -1.5120000000001799 ] ]
    Measured Statistics: {meanExponent=-0.03179922423566625, negative=5, min=-1.5120000000001799, max=-1.5120000000001799, mean=-0.4080000000000133, count=9.0, positive=4, stdDev=1.1474896271619761, zeros=0}
    Feedback Error: [ [ 3.900491041264331E-13, -4.247713292215849E-13, -2.199351811782435E-13 ], [ -7.401856905175919E-13, 4.74953409934642E-13, -2.498001805406602E-13 ], [ 1.2403411631112249E-12, -4.1011638529653283E-13, -1.7985612998927536E-13 ] ]
    Error Statistics: {meanExponent=-12.392608094378751, negative=6, min=-1.7985612998927536E-13, max=-1.7985612998927536E-13, mean=-1.3257913285732078E-14, count=9.0, positive=3, stdDev=5.729266120776919E-13, zeros=0}
    Learning Gradient for weight set 0
    Weights: [ 0.168, 0.592, -1.424, -1.868, 1.252, 1.144, -0.664, -1.36, -1.512 ]
    Implemented Gradient: [ [ 0.364, 0.0, 0.0 ], [ -0.9, 0.0, 0.0 ], [ 1.048, 0.0, 0.0 ], [ 0.0, 0.364, 0.0 ], [ 0.0, -0.9, 0.0 ], [ 0.0, 1.048, 0.0 ], [ 0.0, 0.0, 0.364 ], [ 0.0, 0.0, -0.9 ], [ 0.0, 0.0, 1.048 ] ]
    Implemented Statistics: {meanExponent=-0.15476494142130376, negative=3, min=1.048, max=1.048, mean=0.05688888888888889, count=27.0, positive=6, stdDev=0.4727781825301832, zeros=18}
    Measured Gradient: [ [ 0.36399999999936483, 0.0, 0.0 ], [ -0.8999999999992347, 0.0, 0.0 ], [ 1.0479999999990497, 0.0, 0.0 ], [ 0.0, 0.36399999999936483, 0.0 ], [ 0.0, -0.8999999999992347, 0.0 ], [ 0.0, 1.04800000000016, 0.0 ], [ 0.0, 0.0, 0.36399999999936483 ], [ 0.0, 0.0, -0.900000000000345 ], [ 0.0, 0.0, 1.04800000000016 ] ]
    Measured Statistics: {meanExponent=-0.15476494142164895, negative=3, min=1.04800000000016, max=1.04800000000016, mean=0.05688888888883887, count=27.0, positive=6, stdDev=0.47277818252999954, zeros=18}
    Gradient Error: [ [ -6.351585923880521E-13, 0.0, 0.0 ], [ 7.652767308741204E-13, 0.0, 0.0 ], [ -9.50350909079134E-13, 0.0, 0.0 ], [ 0.0, -6.351585923880521E-13, 0.0 ], [ 0.0, 7.652767308741204E-13, 0.0 ], [ 0.0, 1.5987211554602254E-13, 0.0 ], [ 0.0, 0.0, -6.351585923880521E-13 ], [ 0.0, 0.0, -3.4494629375103614E-13 ], [ 0.0, 0.0, 1.5987211554602254E-13 ] ]
    Error Statistics: {meanExponent=-12.322281725479003, negative=5, min=1.5987211554602254E-13, max=1.5987211554602254E-13, mean=-5.0017603227927424E-14, count=27.0, positive=4, stdDev=3.541971467994917E-13, zeros=18}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.6059e-13 +- 3.3192e-13 [0.0000e+00 - 1.2403e-12] (36#)
    relativeTol: 4.0480e-13 +- 3.3317e-13 [5.9476e-14 - 1.1609e-12] (18#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=2.6059e-13 +- 3.3192e-13 [0.0000e+00 - 1.2403e-12] (36#), relativeTol=4.0480e-13 +- 3.3317e-13 [5.9476e-14 - 1.1609e-12] (18#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.16 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 6.2272 +- 1.1670 [4.9472 - 11.4647]
    Learning performance: 3.6635 +- 0.3940 [3.0778 - 6.1641]
    
```

