# PipelineNetwork
## AsymmetricExplodedTest
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
      "class": "com.simiacryptus.mindseye.network.PipelineNetwork",
      "id": "a864e734-2f23-44db-97c1-5040000008f4",
      "isFrozen": false,
      "name": "PipelineNetwork/a864e734-2f23-44db-97c1-5040000008f4",
      "inputs": [
        "9b02f092-d3dd-43f3-86c5-2917fdc5ea47"
      ],
      "nodes": {
        "a6d00593-09ac-4440-958e-1f47781cc7fc": "a864e734-2f23-44db-97c1-5040000008f6",
        "01babafa-f773-4138-aeb1-70d9434e5f9f": "a864e734-2f23-44db-97c1-5040000008f7",
        "ad76c76f-9b68-465a-a086-a45f230a8a61": "a864e734-2f23-44db-97c1-5040000008f5"
      },
      "layers": {
        "a864e734-2f23-44db-97c1-5040000008f6": {
          "class": "com.simiacryptus.mindseye.layers.cudnn.f64.SimpleConvolutionLayer",
          "id": "a864e734-2f23-44db-97c1-5040000008f6",
          "isFrozen": false,
          "name": "SimpleConvolutionLayer/a864e734-2f23-44db-97c1-5040000008f6",
          "filter": {
            "dimensions": [
              1,
              1,
              4
            ],
            "data": [
              -0.968,
              0.852,
              -0.588,
              0.116
            ]
          },
          "strideX": 1,
          "strideY": 1,
          "simple": false
        },
        "a864e734-2f23-44db-97c1-5040000008f7": {
          "class": "com.simiacryptus.mindseye.layers.cudnn.f64.SimpleConvolutionLayer",
          "id": "a864e734-2f23-44db-97c1-5040000008f7",
          "isFrozen": false,
          "name": "SimpleConvolutionLayer/a864e734-2f23-44db-97c1-5040000008f7",
          "filter": {
            "dimensions": [
              1,
              1,
              4
            ],
            "data": [
              1.236,
              0.888,
              -0.5,
              -1.916
            ]
          },
          "strideX": 1,
          "strideY": 1,
          "simple": false
        },
        "a864e734-2f23-44db-97c1-5040000008f5": {
          "class": "com.simiacryptus.mindseye.layers.cudnn.f64.ImgConcatLayer",
          "id": "a864e734-2f23-44db-97c1-5040000008f5",
          "isFrozen": false,
          "name": "ImgConcatLayer/a864e734-2f23-44db-97c1-5040000008f5",
          "maxBands": 3
        }
      },
      "links": {
        "a6d00593-09ac-4440-958e-1f47781cc7fc": [
          "9b02f092-d3dd-43f3-86c5-2917fdc5ea47"
        ],
        "01babafa-f773-4138-aeb1-70d9434e5f9f": [
          "9b02f092-d3dd-43f3-86c5-2917fdc5ea47"
        ],
        "ad76c76f-9b68-465a-a086-a45f230a8a61": [
          "a6d00593-09ac-4440-958e-1f47781cc7fc",
          "01babafa-f773-4138-aeb1-70d9434e5f9f"
        ]
      },
      "labels": {},
      "head": "ad76c76f-9b68-465a-a086-a45f230a8a61"
    }
```



### Network Diagram
Code from [LayerTestBase.java:94](../../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L94) executed in 0.14 seconds: 
```java
    return Graphviz.fromGraph(toGraph((DAGNetwork) layer))
      .height(400).width(600).render(Format.PNG).toImage();
```

Returns: 

![Result](etc/test.1.png)



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
    	[ [ 0.856, 0.096 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -0.746816, -0.492192, 1.143264 ] ]
    ]
```



### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.05 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (50#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (50#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.07 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 0.856, 0.096 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.5426275011416392, negative=0, min=0.096, max=0.096, mean=0.476, count=2.0, positive=2, stdDev=0.38, zeros=0}
    Output: [
    	[ [ -0.746816, -0.492192, 1.143264 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.12550176913149388, negative=2, min=1.143264, max=1.143264, mean=-0.03191466666666668, count=3.0, positive=1, stdDev=0.8374532899552601, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 0.856, 0.096 ] ]
    ]
    Value Statistics: {meanExponent=-0.5426275011416392, negative=0, min=0.096, max=0.096, mean=0.476, count=2.0, positive=2, stdDev=0.38, zeros=0}
    Implemented Feedback: [ [ -0.968, -0.588, 1.236 ], [ 0.852, 0.116, 0.888 ] ]
    Implemented Statistics: {meanExponent=-0.20156971601507515, negative=2, min=0.888, max=0.888, mean=0.256, count=6.0, positive=4, stdDev=0.8108793580634134, zeros=0}
    Measured Feedback: [ [ -0.9679999999989697, -0.5879999999991448, 1.2359999999977944 ], [ 0.8520000000000749, 0.11600000000000499, 0.8880000000011101 ] ]
    Measured Statistics: {meanExponent=-0.20156971601528673, negative=2, min=0.8880000000011101, max=0.8880000000011101, mean=0.256000000000145, count=6.0, positive=4, stdDev=0.8108793580627148, zeros=0}
    Feedback Error: [ [ 1.0302869668521453E-12, 8.552047958687581E-13, -2.205569060720336E-12 ], [ 7.494005416219807E-14, 4.98212582300539E-15, 1.110112002322694E-12 ] ]
    Error Statistics: {meanExponent=-12.515659252188774, negative=1, min=1.110112002322694E-12, max=1.110112002322694E-12, mean=1.4499281405141082E-13, count=6.0, positive=5, stdDev=1.1379259900352535E-12, zeros=0}
    Learning Gradient for weight set 0
    Weights: [ -0.968, 0.852, -0.588, 0.116 ]
    Implemented Gradient: [ [ 0.856, 0.0, 0.0 ], [ 0.096, 0.0, 0.0 ], [ 0.0, 0.856, 0.0 ], [ 0.0, 0.096, 0.0 ] ]
    Implemented Statistics: {meanExponent=-0.5426275011416392, negative=0, min=0.0, max=0.0, mean=0.15866666666666665, count=12.0, positive=4, stdDev=0.3138208972575847, zeros=8}
    Measured Gradient: [ [ 0.8560000000001899, 0.0, 0.0 ], [ 0.0960000000005401, 0.0, 0.0 ], [ 0.0, 0.8560000000001899, 0.0 ], [ 0.0, 0.09599999999998499, 0.0 ] ]
    Measured Statistics: {meanExponent=-0.5426275011409971, negative=0, min=0.0, max=0.0, mean=0.1586666666667421, count=12.0, positive=4, stdDev=0.31382089725764634, zeros=8}
    Gradient Error: [ [ 1.8995915951336428E-13, 0.0, 0.0 ], [ 5.40095745904523E-13, 0.0, 0.0 ], [ 0.0, 1.8995915951336428E-13, 0.0 ], [ 0.0, -1.5015766408055242E-14, 0.0 ] ]
    Error Statistics: {meanExponent=-12.883415315290804, negative=1, min=0.0, max=0.0, mean=7.541652487693303E-14, count=12.0, positive=3, stdDev=1.570153988883668E-13, zeros=8}
    Learning Gradient for weight set 1
    Weights: [ 1.236, 0.888, -0.5, -1.916 ]
    Implemented Gradient: [ [ 0.0, 0.0, 0.856 ], [ 0.0, 0.0, 0.096 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ] ]
    Implemented Statistics: {meanExponent=-0.5426275011416392, negative=0, min=0.0, max=0.0, mean=0.07933333333333333, count=12.0, positive=2, stdDev=0.23565983016958056, zeros=10}
    Measured Gradient: [ [ 0.0, 0.0, 0.8559999999979695 ], [ 0.0, 0.0, 0.0960000000005401 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ] ]
    Measured Statistics: {meanExponent=-0.5426275011409326, negative=0, min=0.0, max=0.0, mean=0.07933333333320913, count=12.0, positive=2, stdDev=0.23565983016902609, zeros=10}
    Gradient Error: [ [ 0.0, 0.0, -2.0304868897369488E-12 ], [ 0.0, 0.0, 5.40095745904523E-13 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ] ]
    Error Statistics: {meanExponent=-11.979964526864507, negative=1, min=0.0, max=0.0, mean=-1.2419926198603548E-13, count=12.0, positive=1, stdDev=5.936802551320086E-13, zeros=10}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.9289e-13 +- 5.8025e-13 [0.0000e+00 - 2.2056e-12] (30#)
    relativeTol: 8.2952e-13 +- 9.5774e-13 [2.1475e-14 - 2.8130e-12] (12#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=2.9289e-13 +- 5.8025e-13 [0.0000e+00 - 2.2056e-12] (30#), relativeTol=8.2952e-13 +- 9.5774e-13 [2.1475e-14 - 2.8130e-12] (12#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.60 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 17.5281 +- 2.8715 [14.7847 - 31.8264]
    Learning performance: 24.1964 +- 128.4205 [8.4582 - 1301.8206]
    
```

