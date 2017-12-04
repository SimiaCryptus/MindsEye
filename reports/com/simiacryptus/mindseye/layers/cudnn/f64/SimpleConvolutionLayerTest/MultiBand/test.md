# SimpleConvolutionLayer
## MultiBand
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
      "id": "370a9587-74a1-4959-b406-fa4500002b6f",
      "isFrozen": false,
      "name": "SimpleConvolutionLayer/370a9587-74a1-4959-b406-fa4500002b6f",
      "filter": {
        "dimensions": [
          1,
          1,
          9
        ],
        "data": [
          -0.084,
          -1.216,
          1.668,
          0.768,
          0.776,
          0.96,
          0.812,
          0.464,
          -0.336
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
    	[ [ 0.148, 1.316, 1.696 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.2162399999999998, 2.76304, 0.1609440000000001 ] ]
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
      "id": "370a9587-74a1-4959-b406-fa4500002b70",
      "isFrozen": false,
      "name": "ConvolutionLayer/370a9587-74a1-4959-b406-fa4500002b70",
      "filter": {
        "dimensions": [
          1,
          1,
          9
        ],
        "data": [
          -0.084,
          0.768,
          0.812,
          -1.216,
          0.776,
          0.464,
          1.668,
          0.96,
          -0.336
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
    	[ [ 0.148, 1.316, 1.696 ] ]
    ]
    Error: [
    	[ [ 0.0, 0.0, 0.0 ] ]
    ]
    Accuracy:
    absoluteTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (3#)
    relativeTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (3#)
    
```

### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.02 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (60#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (60#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.02 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 0.148, 1.316, 1.696 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.16035218246880364, negative=0, min=1.696, max=1.696, mean=1.0533333333333335, count=3.0, positive=3, stdDev=0.6586963556061993, zeros=0}
    Output: [
    	[ [ 1.2162399999999998, 2.76304, 0.1609440000000001 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.08897291818016606, negative=0, min=0.1609440000000001, max=0.1609440000000001, mean=1.3800746666666666, count=3.0, positive=3, stdDev=1.0685994718338372, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 0.148, 1.316, 1.696 ] ]
    ]
    Value Statistics: {meanExponent=-0.16035218246880364, negative=0, min=1.696, max=1.696, mean=1.0533333333333335, count=3.0, positive=3, stdDev=0.6586963556061993, zeros=0}
    Implemented Feedback: [ [ -0.084, 0.768, 0.812 ], [ -1.216, 0.776, 0.464 ], [ 1.668, 0.96, -0.336 ] ]
    Implemented Statistics: {meanExponent=-0.2120759590205015, negative=3, min=-0.336, max=-0.336, mean=0.42355555555555563, count=9.0, positive=6, stdDev=0.799616451266357, zeros=0}
    Measured Feedback: [ [ -0.08400000000019503, 0.7679999999998799, 0.8119999999989247 ], [ -1.2159999999994398, 0.7760000000001099, 0.46399999999890973 ], [ 1.6679999999991146, 0.9599999999965192, -0.33599999999994745 ] ]
    Measured Statistics: {meanExponent=-0.2120759590207978, negative=3, min=-0.33599999999994745, max=-0.33599999999994745, mean=0.4235555555548751, count=9.0, positive=6, stdDev=0.7996164512657606, zeros=0}
    Feedback Error: [ [ -1.9502455206321656E-13, -1.2012613126444194E-13, -1.0753620216519266E-12 ], [ 5.60218538225854E-13, 1.099120794378905E-13, -1.090294521333135E-12 ], [ -8.852918398360998E-13, -3.4807712268047908E-12, 5.256906021600116E-14 ] ]
    Error Statistics: {meanExponent=-12.395808849008585, negative=6, min=5.256906021600116E-14, max=5.256906021600116E-14, mean=-6.804634016748739E-13, count=9.0, positive=3, stdDev=1.1283486198823042E-12, zeros=0}
    Learning Gradient for weight set 0
    Weights: [ -0.084, -1.216, 1.668, 0.768, 0.776, 0.96, 0.812, 0.464, -0.336 ]
    Implemented Gradient: [ [ 0.148, 0.0, 0.0 ], [ 1.316, 0.0, 0.0 ], [ 1.696, 0.0, 0.0 ], [ 0.0, 0.148, 0.0 ], [ 0.0, 1.316, 0.0 ], [ 0.0, 1.696, 0.0 ], [ 0.0, 0.0, 0.148 ], [ 0.0, 0.0, 1.316 ], [ 0.0, 0.0, 1.696 ] ]
    Implemented Statistics: {meanExponent=-0.16035218246880364, negative=0, min=1.696, max=1.696, mean=0.35111111111111115, count=27.0, positive=9, stdDev=0.6254478296823173, zeros=18}
    Measured Gradient: [ [ 0.14799999999981495, 0.0, 0.0 ], [ 1.3160000000000949, 0.0, 0.0 ], [ 1.6959999999999198, 0.0, 0.0 ], [ 0.0, 0.1479999999975945, 0.0 ], [ 0.0, 1.3160000000000949, 0.0 ], [ 0.0, 1.6959999999999198, 0.0 ], [ 0.0, 0.0, 0.14799999999981495 ], [ 0.0, 0.0, 1.3159999999989846 ], [ 0.0, 0.0, 1.6959999999999198 ] ]
    Measured Statistics: {meanExponent=-0.1603521824697457, negative=0, min=1.6959999999999198, max=1.6959999999999198, mean=0.3511111111109688, count=27.0, positive=9, stdDev=0.6254478296822843, zeros=18}
    Gradient Error: [ [ -1.8504642262939797E-13, 0.0, 0.0 ], [ 9.481304630298837E-14, 0.0, 0.0 ], [ -8.01581023779363E-14, 0.0, 0.0 ], [ 0.0, -2.405492471879711E-12, 0.0 ], [ 0.0, 9.481304630298837E-14, 0.0 ], [ 0.0, -8.01581023779363E-14, 0.0 ], [ 0.0, 0.0, -1.8504642262939797E-13 ], [ 0.0, 0.0, -1.0154099783221682E-12 ], [ 0.0, 0.0, -8.01581023779363E-14 ] ]
    Error Statistics: {meanExponent=-12.71244607785654, negative=7, min=-8.01581023779363E-14, max=-8.01581023779363E-14, mean=-1.4229050036994472E-13, count=27.0, positive=2, stdDev=4.859710770867754E-13, zeros=18}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 3.2752e-13 +- 7.2048e-13 [0.0000e+00 - 3.4808e-12] (36#)
    relativeTol: 8.5775e-13 +- 1.8307e-12 [2.3632e-14 - 8.1267e-12] (18#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=3.2752e-13 +- 7.2048e-13 [0.0000e+00 - 3.4808e-12] (36#), relativeTol=8.5775e-13 +- 1.8307e-12 [2.3632e-14 - 8.1267e-12] (18#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.15 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 5.9476 +- 1.4411 [4.8902 - 13.5222]
    Learning performance: 3.5876 +- 0.8565 [3.0920 - 11.6927]
    
```

