{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: __init__

   {% block methods %}
   {% set public_methods = [] %}
   {% for item in methods %}
   {% if item != '__init__' and not item.startswith('_') %}
   {% set _ = public_methods.append(item) %}
   {% endif %}
   {% endfor %}
   {% if public_methods %}
   .. rubric:: Methods

   .. autosummary::
   {% for item in public_methods %}
      ~{{ name }}.{{ item }}
   {% endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% set public_attributes = [] %}
   {% for item in attributes %}
   {% if not item.startswith('_') %}
   {% set _ = public_attributes.append(item) %}
   {% endif %}
   {% endfor %}
   {% if public_attributes %}
   .. rubric:: Attributes

   .. autosummary::
   {% for item in public_attributes %}
      ~{{ name }}.{{ item }}
   {% endfor %}
   {% endif %}
   {% endblock %}