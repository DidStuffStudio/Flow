using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Rotate : MonoBehaviour
{
    [SerializeField] private float speed = 10;
    [SerializeField] private FlowField flowField;
    private Vector3 _positionToLookAround;

    private void Start()
    {
        var x = flowField.Width * flowField.Spacing / 2;
        var y = flowField.Height * flowField.Spacing / 2;
        var z = flowField.Depth * flowField.Spacing / 2;
        _positionToLookAround = new Vector3(x, y, z);
    }

    private void Update()
    {
        
        transform.RotateAround(_positionToLookAround,Vector3.up,speed*Time.deltaTime);
        transform.LookAt(_positionToLookAround);
    }
}
