# CIFO_Project

Report link
https://liveeduisegiunl-my.sharepoint.com/:w:/r/personal/20240549_novaims_unl_pt/Documents/Report%20CIfO%20group%20S.docx?d=waf42e720ba9d48e08dacc689dbee3981&csf=1&web=1&e=JOrHX4

## Wedding Seating Optimization
The goal is to optimize the seating arrangement of **guests across tables** to maximize guest happiness and minimize conflicts based on their relationships.

Each solution represents a complete seating arrangement, specifying which guests are seated at each table. These are the constraints that must be verified in every solution of the search space (no object is considered a solution if it doesn’t comply with these):
- Each guest must be assigned to exactly one table.
- Table placement matters only in terms of grouping. **The arrangement within a table is irrelevant.**

**Impossible Configurations**: Any arrangement where a guest is seated at multiple tables or left unassigned is not part of the search space and is considered impossible. It is forbidden to generate such an arrangement during evolution.

[Here](https://liveeduisegiunl-my.sharepoint.com/:x:/g/personal/imagessi_novaims_unl_pt/EYTlb599xcZIlhLRp_lBUpUBdsxsFGoketsp5AcX1NuUew?e=fvOLyn) you can find a pairwise relationship matrix of all **64 attendees**, who need to be distributed across **8 tables**. You can also consult the [chart](https://liveeduisegiunl-my.sharepoint.com/personal/imagessi_novaims_unl_pt/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fimagessi%5Fnovaims%5Funl%5Fpt%2FDocuments%2Fguests%2Epng&parent=%2Fpersonal%2Fimagessi%5Fnovaims%5Funl%5Fpt%2FDocuments&ga=1) of the guests and their relationships, as well as a [table](https://liveeduisegiunl-my.sharepoint.com/personal/imagessi_novaims_unl_pt/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fimagessi%5Fnovaims%5Funl%5Fpt%2FDocuments%2Frelationships%2Epng&parent=%2Fpersonal%2Fimagessi%5Fnovaims%5Funl%5Fpt%2FDocuments&ga=1) explaining relationship values.

The goal is to maximize the overall relationship score by optimizing guest seatings within each table and then aggregating these scores across all tables.

<img width="611" alt="guests" src="https://github.com/user-attachments/assets/3c559f23-9d97-44d8-a7e9-5e4af3992f4a" />

<img width="631" alt="relationships" src="https://github.com/user-attachments/assets/dad2637e-d0ad-4bef-8cfb-fcf2fdb3b4d1" />


