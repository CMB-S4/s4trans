from collections import OrderedDict

# Create a ordered dictionary, order is important for ingestion
Fd = OrderedDict()

# From file info not in header
# ID is unique per table
Fd['ID'] = 'TEXT'
Fd['SIMID'] = 'TEXT'
Fd['PROJ'] = 'TEXT'
Fd['FRACTION'] = 'FLOAT'
