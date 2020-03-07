from setuptools import setup, find_packages

setup(name='colorme',
      version='0.1',
      description='Color me impressed!',
      packages=find_packages(),
      zip_safe=False,
      install_requries=[
            # TODO pin versions
            "torch>=1.4",
            "torchvision",
            "click",
            "pandas",
            "pillow",
      ],
      entry_points="""
          [console_scripts]
          colorme=colorme.scripts.training:colorme
      """,
      )
