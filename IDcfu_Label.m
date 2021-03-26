%function[] = IDcfu_Label(fileimage, water)
    %Input: day0, plate# and .tif file
    %This script identifies bacteria colonies in a petri dish. It generates a
    %mask first to avoid the outer part of the plate. Later, segmentation and
    %label identification allows to target even those colonies that are not
    %completely circular. It retrieves a .jpg file with the identifies colonies
    %and a .mat file with a label, morphological and RGB-lab pixel
    %information. 
    %Last version 25.05.2020

    %clear

    %to test without function
    fileimage = 'i1_d25_30µl-nr5';
    water = 1;


    %Read files. Batch of pictures from a plate taken along several days 
    %16-bit file, RGB range 0-2500
    I = imread(fileimage, 'tif');
    %Turn into 8-bit, RGB range 0-255 with a simple division
    I = uint8(I/257);

    %%
    %%Remove uninterested region from the image 
    %Get the image size to remove the rim of the petri dish
    imageSize = size(I);
    %center and radius of circle ([c_col, c_row, r]). I set my center at
    %[y0,x0] = [1040, 1015] and r = 845
    ci = [1040, 1015, 845]; 
    %Make a grid the same dimensions as the original image
    [xx,yy] = ndgrid((1:imageSize(1))-ci(1),(1:imageSize(2))-ci(2));
    %Make a mask that will turn black all the area outside the plate by
    %fitting a circle with the size of the plate
    mask = uint8((xx.^2 + yy.^2)<ci(3)^2);
    %Generate the new image, cropped imaged after the mask is applied
    croppedImage = uint8(true(size(I)));
    croppedImage(:,:,1) = I(:,:,1).*mask;
    croppedImage(:,:,2) = I(:,:,2).*mask;
    croppedImage(:,:,3) = I(:,:,3).*mask;

%Remove comments if you want to print the crooped image
%     figure
%     imshow(croppedImage)

%%
    %The red channel was the most informative one, therefore for colony identification
    %I decided to only take information from the red channel
    
    %Correct non-uniform ilumination. Adjust specific range if needed
    rgbI = imadjust(croppedImage(:,:,1), [0 0.60],[]);

%Remove comments if you want to print and save the image after  the crooped image
%     figure
%     imshow(rgbI)
%     print(strcat(filename{1},'-adjust'),'-dpng');
%     close;
    
%%
    %There are two types of colonies on the plates, ones with higher RGB values
    %than the background and some other with lower RGB values than the
    %background. I implemented a two-step process to identify all of them.
    
    %Filter bright colonies
    rgbBW = rgbI >=200;%imshow(rgbBW)
    %remove connected objects
    rgbI_nobord = imclearborder(rgbBW,8);%imshow(rgbI_nobord)
    %to fill up holes
    rgbI_noholes = imfill(rgbI_nobord,'holes');%imshow(rgbI_final)       
    %smooth object. Avoids extracting background information
    seD = strel('diamond',1);
    rgbI_final = imerode(rgbI_noholes,seD);
    
    
    %Find colonies using boundary.
    %B, returns an array of pixel locations
    %L, label matrix of objects
    %n, number of objects (labels)
    [B1,L1,n1] = bwboundaries(rgbI_final,'noholes');   

    %Filter dark colonies
    rgbBW = rgbI < 50;%imshow(rgbBW)
    %remove connected objects
    rgbI_nobord = imclearborder(rgbBW,8);%imshow(rgbI_nobord)
    %rgbI_final = rgbI_nobord;
    %smooth object. Avoids extracting background information.
    seD = strel('diamond',1);
    rgbI_final = imerode(rgbI_nobord,seD);

    %Find colonies using boundary
    [B2,L2,n2] = bwboundaries(rgbI_final,'noholes');

    %Match both boundaries, most of the time no dark colonies will be
    %identify, therefor only the information from brigth colonies will be
    %used.
    if isempty(n2)
        %BW image
        B = B1;
        L = L1;
        n_colonies = n1;
    else
        %BW image
        B = [B1; B2];
        L = (L1 | L2);
        n_colonies = (n1+n2);
    end
%%
if water == 1
%Extra option to do watershed or not.
    %Remove small white objects from the image
    BW = ~bwareaopen(~L, 10);
    %BW = L;
    %Get distance between white objects
    D = -bwdist(~BW);
    %Computes local maxima. Center of the colonies
    mask = imextendedmin(D,1,4);
    
%     figure
%     imshowpair(BW,mask,'blend')
%     print(strcat(file,'-blend'),'-dpng');
%     close;
    
    minima = imimposemin(D,mask);
    Lw = watershed(minima);
    L = BW;
    L(Lw == 0) = 0;
    Ll = bwlabel(L);   

%     figure
%     imshow(BW2)
%     print(strcat(file,'-BW'),'-dpng');
%     close;
else
% Generate labeled BW
    Ll = bwlabel(L);   
end
 
%%    

%Remove comments if one wants to check the boundaries found
%     figure
%     imshow(L)
%     hold on
%     for k = 1:length(B)
%        boundary = B{k};
%        plot(boundary(:,2), boundary(:,1), 'r', 'LineWidth', 2)
%     end
        
%Remove if you want to print and save the BW image got from the boundaries
%function
%     figure
%     imshow(L)
%     print(strcat(filename{1},'-BW'),'-dpng');
%     close;

%%
    %get morphological stats
    stats=  regionprops(L, 'Centroid', 'Area', 'EquivDiameter', 'Eccentricity');
    Centroid = floor(cat(1,stats.Centroid));
    Eccentricity = cat(1,stats.Eccentricity);
    Area = cat(1,stats.Area);
    Diameter = cat(1,stats.EquivDiameter);

    %%
    %Filter 'bad quality data'
    %Filter by area bigger than n and eccentricity, the closes to 1 the more
    %line segment, the closes to 0 the more circular shape
    filter1 = find(Area > 200 & Area < 70000 & Eccentricity < 0.79); 

    %Find the elements close the the petri dish walls
    filter2 = zeros(length(filter1),1);

    %Calculate indexes to plot a circle and compare with the indexes in mask
    th = 0:pi/5:2*pi;
    for i = 1:length(filter1)
        %imshow(croppedImage);
        %Fit a circle
        xunit = floor(Diameter(filter1(i))/2 * cos(th) + Centroid(filter1(i),1));
        yunit = floor(Diameter(filter1(i))/2 * sin(th) + Centroid(filter1(i),2));
        %Find within the boundaries. Check ci variable
        mean_xy = mean(yunit>200 & yunit<1875 & xunit>200 & xunit<1875);
        if mean_xy == 1
           filter2(i) = filter1(i);
        end
    end

    filter2 = nonzeros(filter2);
%%
%find unique labels in the BW images. Should be equal to the number of
%elements in B
e = unique(Ll);
etozero = setdiff(e, filter2);

for i = 1:length(etozero)
    Ll(Ll == etozero(i)) = 0;
end

%Label area inside the colonies as 2
Ll(Ll > 0) = 2;
%Border as 1
outline = double(bwperim(Ll));

%Three labels
labeled = (Ll + outline);
%Get three different colors
Lrgb = label2rgb(labeled, 'spring','c','shuffle');

figure
imshow(Lrgb)
%%
%BW = imbinarize(Ll);
% if water == 1
% %     figure
% %     imshow(Lrgb)
% %     print(strcat(fileimage,'-BWw'), '-dtiff');
% %     close;
%       imwrite(Lrgb, strcat(fileimage,'-Wgt.tiff') )
% else
% %     figure
% %     imshow(Lrgb)
% %     print(strcat(fileimage,'-BW'), '-dtiff');
% %     close;
%     imwrite(Lrgb, strcat(fileimage,'-gt.tiff') )
% end

%end