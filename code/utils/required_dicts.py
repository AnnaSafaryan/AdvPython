"""
Списки требований для разных знаечний аргумента key_arg в check_args
"""

lang_required = {
    'cross': ['direction', 'common_output_vectors_path']
    }

cross_method_required = {'model': ['src_embeddings_path', 'tar_embeddings_path'],
                         'translation': ['tar_embeddings_path', 'bidict_path'],
                         'projection': ['src_embeddings_path', 'tar_embeddings_path',
                                        'projection_path']
                         }

mono_method_required = {'model': ['src_embeddings_path'],
                        'translation': ['tar_embeddings_path', 'bidict_path'],
                        'projection': ['src_embeddings_path', 'tar_embeddings_path',
                                       'projection_path']
                        }

url_required = {
    1: ['url_mapping_path'],
    0: []
    }

included_required = {
    0: ['udpipe_path', 'embeddings_path', 'method'],
    1: []
    }

notincl_method_required = {'model': ['embeddings_path'],
                           'translation': ['embeddings_path', 'bidict_path'],
                           'projection': ['embeddings_path', 'projection_path']
                           }

preprocessed_required = {
    0: ['udpipe_path'],
    1: []
}