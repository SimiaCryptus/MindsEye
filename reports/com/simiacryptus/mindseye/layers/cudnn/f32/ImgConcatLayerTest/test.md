# ImgConcatLayer
## ImgConcatLayerTest
### Json Serialization
Code from [LayerTestBase.java:84](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L84) executed in 0.00 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f32.ImgConcatLayer",
      "id": "c88cbdf1-1c2a-4a5e-b964-8909000002f0",
      "isFrozen": false,
      "name": "ImgConcatLayer/c88cbdf1-1c2a-4a5e-b964-8909000002f0",
      "maxBands": -1
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:121](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L121) executed in 0.00 seconds: 
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
    	[ [ 1.16 ], [ -1.444 ] ],
    	[ [ 0.984 ], [ 1.036 ] ]
    ],
    [
    	[ [ 0.252 ], [ -0.792 ] ],
    	[ [ -1.924 ], [ -1.172 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.159999966621399, 0.25200000405311584 ], [ -1.444000005722046, -0.7919999957084656 ] ],
    	[ [ 0.984000027179718, -1.9240000247955322 ], [ 1.0360000133514404, -1.1720000505447388 ] ]
    ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.02 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Component: ImgConcatLayer/c88cbdf1-1c2a-4a5e-b964-8909000002f0
    Inputs: [
    	[ [ 1.16 ], [ -1.444 ] ],
    	[ [ 0.984 ], [ 1.036 ] ]
    ],
    [
    	[ [ 0.252 ], [ -0.792 ] ],
    	[ [ -1.924 ], [ -1.172 ] ]
    ]
    output=[
    	[ [ 1.159999966621399, 0.25200000405311584 ], [ -1.444000005722046, -0.7919999957084656 ] ],
    	[ [ 0.984000027179718, -1.9240000247955322 ], [ 1.0360000133514404, -1.1720000505447388 ] ]
    ]
    measured/actual: [ [ 1.0001659393310547, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.9995698928833008, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 1.0001659393310547, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 1.0001659393310547, 0.0, 0.0, 0.0, 0.0 ] ]
    implemented/expected: [ [ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0 ] ]
    error: [ [ 1.659393310546875E-4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, -4.3010711669921875E-4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 1.659393310546875E-4, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 1.659393310546875E-4, 0.0, 0.0, 0.0, 0.0 ] ]
    Component: ImgConcatLayer/c88cbdf1-1c2a-4a5e-b964-8909000002f0
    Inputs: [
    	[ [ 1.16 ], [ -1.444 ] ],
    	[ [ 0.984 ], [ 1.036 ] ]
    ],
    [
    	[ [ 0.252 ], [ -0.792 ] ],
    	[ [ -1.924 ], [ -1.172 ] ]
    ]
    output=[
    	[ [ 1.159999966621399, 0.25200000405311584 ], [ -1.444000005722046, -0.7919999957084656 ] ],
    	[ [ 0.984000027179718, -1.9240000247955322 ], [ 1.0360000133514404, -1.1720000505447388 ] ]
    ]
    measured/actual: [ [ 0.0, 0.0, 0.0, 0.0, 0.9998679161071777, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.0001659393310547, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0001659393310547, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0001659393310547 ] ]
    implemented/expected: [ [ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 ] ]
    error: [ [ 0.0, 0.0, 0.0, 0.0, -1.3208389282226562E-4, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.659393310546875E-4, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.659393310546875E-4, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.659393310546875E-4 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.4341e-05 +- 7.1778e-05 [0.0000e+00 - 4.3011e-04] (64#)
    relativeTol: 9.7365e-05 +- 4.4843e-05 [6.6046e-05 - 2.1510e-04] (8#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.07 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 2.5700 +- 0.4242 [2.3083 - 5.0441]
    Learning performance: 1.2203 +- 0.1626 [1.0943 - 2.5819]
    
```

