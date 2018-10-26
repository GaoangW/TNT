function docNode = csv_pascal(positionMat, csv_file, img, class_table)

docNode = com.mathworks.xml.XMLUtils.createDocument('annotation');
annotation = docNode.getDocumentElement;

% folder
folder = docNode.createElement('folder');
folder.appendChild(docNode.createTextNode('VOC2007'));
annotation.appendChild(folder);

% filename
csv_name = csv_file;
dotLocation = find(csv_name=='.');
dotLocation = dotLocation(end);
imgName = strcat(csv_name(1:dotLocation),'jpg');
filePath = csv_name(1:dotLocation-1);

filename = docNode.createElement('filename');
filename.appendChild(docNode.createTextNode(imgName));
annotation.appendChild(filename);

% source
source = docNode.createElement('source');
annotation.appendChild(source);
database = docNode.createElement('database');
database.appendChild(docNode.createTextNode('UA-DETRAC'));
source.appendChild(database);
childAnnotation = docNode.createElement('annotation');
childAnnotation.appendChild(docNode.createTextNode('UWEE 2018'));
source.appendChild(childAnnotation);
image = docNode.createElement('image');
image.appendChild(docNode.createTextNode('UA-DETRAC Image'));
source.appendChild(image);

% size
img_size = size(img);
sizeImage = docNode.createElement('size');
annotation.appendChild(sizeImage);
width = docNode.createElement('width');
width.appendChild(docNode.createTextNode(num2str(img_size(2))));
sizeImage.appendChild(width);
height = docNode.createElement('height');
height.appendChild(docNode.createTextNode(num2str(img_size(1))));
sizeImage.appendChild(height);
depth = docNode.createElement('depth');
depth.appendChild(docNode.createTextNode(num2str(img_size(3))));
sizeImage.appendChild(depth);

% segment
segmented = docNode.createElement('segmented');
segmented.appendChild(docNode.createTextNode('0'));
annotation.appendChild(segmented);

% object
num = size(positionMat,1);
numObject = 1;
while numObject <= num
    object{numObject} = docNode.createElement('object');
    annotation.appendChild(object{numObject});
    name{numObject} = docNode.createElement('name');
    class = class_table{positionMat(numObject, 7)};
    name{numObject}.appendChild(docNode.createTextNode(class));
    object{numObject}.appendChild(name{numObject});
    pose{numObject} = docNode.createElement('pose');
    pose{numObject}.appendChild(docNode.createTextNode('Unspecified'));
    object{numObject}.appendChild(pose{numObject});
    truncated{numObject} = docNode.createElement('truncated');
    truncated{numObject}.appendChild(docNode.createTextNode('0'));
    object{numObject}.appendChild(truncated{numObject});
    difficult{numObject} = docNode.createElement('difficult');
    difficult{numObject}.appendChild(docNode.createTextNode('0'));
    object{numObject}.appendChild(difficult{numObject});
    bndbox{numObject} = docNode.createElement('bndbox');
    object{numObject}.appendChild(bndbox{numObject});
    xmin{numObject} = docNode.createElement('xmin');
    str = num2str(positionMat(numObject,2));
    xmin{numObject}.appendChild(docNode.createTextNode(str));
    bndbox{numObject}.appendChild(xmin{numObject});
    ymin{numObject} = docNode.createElement('ymin');
    str = num2str(positionMat(numObject,3));
    ymin{numObject}.appendChild(docNode.createTextNode(str));
    bndbox{numObject}.appendChild(ymin{numObject});
    xmax{numObject} = docNode.createElement('xmax');
    str = num2str(positionMat(numObject,4)+positionMat(numObject,2));
    xmax{numObject}.appendChild(docNode.createTextNode(str));
    bndbox{numObject}.appendChild(xmax{numObject});
    ymax{numObject} = docNode.createElement('ymax');
    str = num2str(positionMat(numObject,5)+positionMat(numObject,3));
    ymax{numObject}.appendChild(docNode.createTextNode(str));
    bndbox{numObject}.appendChild(ymax{numObject});
    numObject = numObject+1;

end

