from operator import itemgetter
from tree_format import format_tree


def print_tree(tree):
  print (format_tree(tree, format_node=itemgetter(0), get_children=itemgetter(1)))


project_tree = (
    'project/', [
      ('system/', [
        ('rdf_dir/', [
            ('mol1_mol1.xvg', []),
            ('mol1_mol2.xvg', []),
            ('mol1_mol2.xvg', []),
          ]),
        ('system_npt.edr', []),
        ('system.top', []),
        ],
    )
  ]
)

pc_tree = (
    'pure_components/', [
      ('molecule_temp/', [
        ('molecule_temp_npt.edr', []),
        ('molecule_temp.top', []),
        ],
    )
  ]
)

print_tree(pc_tree)