using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UIElements;
using Random = UnityEngine.Random;

public class FlowParticle : MonoBehaviour
{
    private Vector3 _velocity;
    private Vector3 _acceleration;
    private int _minX, _maxX, _minY, _maxY, _minZ, _maxZ;
    private Material mat;
    [SerializeField] private List<Color> colours;
    [SerializeField] private float maxSpeed = 4;


    public Vector3 CurrentPos { get; set; }

    public Vector3 Velocity
    {
        get => _velocity;
        set
        {
            if (!AcceptNewPosition) return;
            if (value.magnitude > maxSpeed)
                value = value.normalized * maxSpeed;
            _velocity = value;
        }
    }

    public Vector3 Acceleration
    {
        get => _acceleration;
        set { if(AcceptNewPosition) _acceleration = value; }
    }

    public FlowField Field { get; set; }

    public bool AcceptNewPosition { get; set; } = true;


    /*private void Update()
    {
        if(AcceptNewPosition)
            CalculatePosition();
    }*/

    public void SetBounds()
    {
       
        var tr = GetComponent<TrailRenderer>();
        var col = colours[Random.Range(0, colours.Count)];
        tr.startColor = col;
        tr.endColor = col;
        tr.material.color = col;
        
        _minX = 0;
        _minY = 0;
        _minZ = 0;

        _maxX = Field.Width*(int)(Field.Spacing)-1;
        _maxY = Field.Height*(int)(Field.Spacing)-1;
        _maxZ = Field.Depth*(int)(Field.Spacing)-1;
    }

    private void CalculatePosition()
    {
        var x = Mathf.FloorToInt(CurrentPos.x);
        var y = Mathf.FloorToInt(CurrentPos.y);
        var z = Mathf.FloorToInt(CurrentPos.z);
        
    var id = x + y*(Field.Width/(int)Field.Spacing) + z*(Field.Width/(int)Field.Spacing)*(Field.Height/(int)Field.Spacing);
    Acceleration = Field._vectors[id].normalized;
    Velocity += _acceleration;
    SetPosition(CurrentPos + _velocity);
    _acceleration = Vector3.zero;
    }

public void SetPosition(Vector3 pos)
{
  var flagMoved = false;
  var newPos = pos;
  if (pos.x < _minX)
  {
      newPos.x = _maxX;
      flagMoved = true;
  }
  else if (pos.x > _maxX)
  {
      newPos.x = _minX;
      flagMoved = true;
  }

  if (pos.y < _minY)
  {
      newPos.y = _maxY;
      flagMoved = true;
  }
  else if (pos.y > _maxY)
  {
      newPos.y = _minY;
      flagMoved = true;
  }

  if (pos.z < _minZ)
  {
      newPos.z = _maxZ;
      flagMoved = true;
  }
  else if (pos.z > _maxZ)
  {
      newPos.z = _minZ;
      flagMoved = true;
  }

  if (flagMoved)
  {
      AcceptNewPosition = false;
      CreateNewParticle(newPos);
  }

  if (!AcceptNewPosition) return;
  transform.position = newPos;
  CurrentPos = newPos;
  
}

private void CreateNewParticle(Vector3 pos)
{
  var newGO = Instantiate(gameObject);
  var p =newGO.GetComponent<FlowParticle>();
  Field.SwitchParticle(this, p);
  p.Field = Field;
  p.SetBounds();
  p.Acceleration = _acceleration;
  p.Velocity = _velocity;
  p.SetPosition(pos);
}
}
