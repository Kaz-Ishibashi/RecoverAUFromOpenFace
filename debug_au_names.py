# debug_au_names.py
# Debug AU name ordering
# AU名の順序をデバッグ

# The issue: CP6 predictions match, but CP7 offsets differ
# This suggests the cutoff lookup is using wrong AU-to-cutoff mapping

# OpenFace iteration order: std::map<string, vector<double>> AU_predictions_reg_all_hist
# This is alphabetically sorted by AU name

# RecoverAU sorted order should be: sorted(GetAURegNames())
# GetAURegNames() = static AU names + dynamic AU names

# Let's figure out what AU names are likely

# Typical OpenFace AU names for regression (intensity):
# Static: AU01, AU02, AU04, AU05, AU06, AU07, AU09, AU10, AU12, AU14, AU15, AU17, AU20, AU23, AU25, AU26, AU45
# These are loaded from AU_all_best.txt which combines static and dynamic models

# Looking at previous debug output:
# CP7 GT/TG both have 17 entries (indices 0-16)
# Indices 2, 4, 5, 7, 8, 9 have offset=0.0 (matching)
# These are likely static AUs (no cutoff) or AUs not in dyn_au_names

print("AU Ordering Analysis")
print("=" * 70)

# In OpenFace, the map keys (AU names) are sorted alphabetically
# So the order would be:
# AU01, AU02, AU04, AU05, AU06, AU07, AU09, AU10, AU12, AU14, AU15, AU17, AU20, AU23, AU25, AU26, AU45
# (17 AUs in alphabetical order)

# In RecoverAU, GetAURegNames() returns static first, then dynamic
# Let's assume static AUs are: AU01, AU02, AU04, AU05, AU06, AU12
# And dynamic AUs are: AU07, AU09, AU10, AU14, AU15, AU17, AU20, AU23, AU25, AU26, AU45
# (This is just a guess based on typical OpenFace configuration)

# The key issue: The CUTOFF for each AU must be looked up correctly
# In OpenFace:
#   - Iterates through AU names in alphabetical order
#   - For each AU, checks if it's in dyn_au_names
#   - If yes, uses the cutoff at that index in dyn_au_names

# In RecoverAU:
#   - Same logic, but the data comes from different structures

# The CP6 passing means raw predictions are correct
# The CP7 failing means offset calculation differs

# The most likely issue:
# The cutoff ratios might be associated with different AUs between GT and TG

print("""
Key insight:
- CP6 (raw predictions) match PERFECTLY
- CP7 (offsets) differ

Since offset = predictions[(size * cutoff_ratio)], and predictions are identical,
the only way offsets can differ is if:
1. Different cutoff_ratio is being used
2. Different predictions are being collected (unlikely since CP6 matched)

The cutoff_ratio is looked up by finding the AU name in dyn_au_names.
If the AU name at sorted index X in RecoverAU is different from the AU name
at sorted index X in OpenFace's map, the cutoff lookup will fail.

Let me check if there's a case sensitivity or naming difference...
""")
