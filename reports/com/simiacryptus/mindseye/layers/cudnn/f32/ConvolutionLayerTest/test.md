# ConvolutionLayer
## ConvolutionLayerTest
### Json Serialization
Code from [LayerTestBase.java:84](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L84) executed in 0.05 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f32.ConvolutionLayer",
      "id": "952a7607-5593-4ecf-8a1f-3e0900000001",
      "isFrozen": false,
      "name": "ConvolutionLayer/952a7607-5593-4ecf-8a1f-3e0900000001",
      "filter": {
        "dimensions": [
          1,
          1,
          4
        ],
        "data": [
          1.044,
          -0.136,
          -1.1,
          -1.656
        ]
      },
      "strideX": 1,
      "strideY": 1
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:121](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L121) executed in 0.02 seconds: 
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
    	[ [ -1.776, -0.28 ], [ -0.94, 0.576 ], [ -0.084, 1.56 ] ],
    	[ [ 0.488, 0.024 ], [ 0.488, -1.312 ], [ 0.52, -0.648 ] ],
    	[ [ 1.104, 1.204 ], [ -0.888, 1.68 ], [ 1.192, -0.012 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -1.5461441278457642, 0.7052160501480103 ], [ -1.614959955215454, -0.8260159492492676 ], [ -1.8036959171295166, -2.5719358921051025 ] ],
    	[ [ 0.4830720126628876, -0.10611200332641602 ], [ 1.9526721239089966, 2.106304168701172 ], [ 1.255679965019226, 1.0023679733276367 ] ],
    	[ [ -0.17182405292987823, -2.143968105316162 ], [ -2.7750720977783203, -2.6613118648529053 ], [ 1.2576481103897095, -0.1422400176525116 ] ]
    ]
```



### Reference Implementation
Code from [LayerTestBase.java:132](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L132) executed in 0.72 seconds: 
```java
    System.out.println(new GsonBuilder().setPrettyPrinting().create().toJson(referenceLayer.getJson()));
    getEquivalencyTester().test(referenceLayer, layer, inputPrototype);
```
Logging: 
```
    {
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "952a7607-5593-4ecf-8a1f-3e0900000009",
      "isFrozen": false,
      "name": "ConvolutionLayer/952a7607-5593-4ecf-8a1f-3e0900000009",
      "filter": {
        "dimensions": [
          1,
          1,
          4
        ],
        "data": [
          1.044,
          -0.136,
          -1.1,
          -1.656
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
    	[ [ -1.776, -0.28 ], [ -0.94, 0.576 ], [ -0.084, 1.56 ] ],
    	[ [ 0.488, 0.024 ], [ 0.488, -1.312 ], [ 0.52, -0.648 ] ],
    	[ [ 1.104, 1.204 ], [ -0.888, 1.68 ], [ 1.192, -0.012 ] ]
    ]
    Error: [
    	[ [ -1.2784576419733185E-7, 5.014801018887738E-8 ], [ 4.478454584955216E-8, 5.075073228333338E-8 ], [ 8.287048358646132E-8, 1.0789489746088066E-7 ] ],
    	[ [ 1.266288751633482E-8, -3.3264160176349478E-9 ], [ 1.2390899639669328E-7, 1.6870117169887067E-7 ], [ -3.498077405517108E-8, -2.6672363206969862E-8 ] ],
    	[ [ -5.2929878230356664E-8, -1.0531616245756936E-7 ], [ -9.777832010726684E-8, 1.3514709440443085E-7 ], [ 1.103897093734929E-7, -1.76525115913595E-8 ] ]
    ]
    Accuracy:
    absoluteTol: 7.5209e-08 +- 4.7300e-08 [3.3264e-09 - 1.6870e-07] (18#)
    relativeTol: 3.4486e-08 +- 3.1734e-08 [1.3107e-08 - 1.5402e-07] (18#)
    
```

### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.20 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 8.3296e-05 +- 2.5616e-04 [0.0000e+00 - 1.6863e-03] (396#)
    relativeTol: 5.1261e-04 +- 8.3270e-04 [1.6798e-06 - 3.7868e-03] (72#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.33 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 10.9151 +- 3.6417 [7.3496 - 33.7757]
    Learning performance: 10.2784 +- 3.9188 [5.7366 - 20.6296]
    
```

